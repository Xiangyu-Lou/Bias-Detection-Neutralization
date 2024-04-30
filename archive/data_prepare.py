import pandas as pd
import json
from sklearn.model_selection import train_test_split

try:
    data = pd.read_csv(
        'data/biased.full',
        sep='\t',
        quotechar='"',
        names=["id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_POS_tags", "tgt_parse_tags"],
        on_bad_lines='skip'
    )
except Exception as e:
    print(f"An error occurred: {e}")


sampled_data = data.sample(n=2000, random_state=42)

data_with_label_1 = sampled_data.iloc[:1000]
data_with_label_0 = sampled_data.iloc[1000:]

df_label_1 = pd.DataFrame({
    'label': 1,
    'source_text': data_with_label_1['src_raw'],
    'target_text': data_with_label_1['tgt_raw']
})

df_label_0 = pd.DataFrame({
    'label': 0,
    'source_text': data_with_label_0['tgt_raw'],
    'target_text': [''] * 1000
})

final_df = pd.concat([df_label_1, df_label_0])
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv('data/sampled_data.csv', index=False)

train_data, test_data = train_test_split(final_df, test_size=0.4, random_state=42)

train_data = train_data.reset_index(drop=True)
train_data.insert(0, 'Sample Index', range(1, len(train_data) + 1))
test_data = test_data.reset_index(drop=True)
test_data.insert(0, 'Sample Index', range(1, len(test_data) + 1))

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

# chat-completion structure
def format_chat_completion(source_text, label, target_text=None):
    response = [
        {"role": "system", "content": "You are an assistant trained to identify and neutralize subjective bias. If you detect subjective bias, respond with 'This is subjective bias text.' and provide the neutralized text. If no subjective bias is detected, respond with 'This text does not contain detectable subjective bias."},
        {"role": "user", "content": source_text},
    ]
    if label == 1:
        assistant_response = "This is subjective bias text."
        if target_text:
            assistant_response += f" The neutralized text is: {target_text}"
    else:
        assistant_response = "This text does not contain detectable subjective bias."
    response.append({"role": "assistant", "content": assistant_response})
    return response

def write_to_jsonl(data, file_name):
    with open(file_name, 'w') as jsonl_file:
        for index, row in data.iterrows():
            chat_completion = format_chat_completion(row['source_text'], row['label'], row.get('target_text', None))
            entry = {
                "messages": chat_completion
            }
            jsonl_file.write(json.dumps(entry) + '\n')

write_to_jsonl(train_data, 'data/fine_tuning_train_data.jsonl')
write_to_jsonl(test_data, 'data/fine_tuning_test_data.jsonl')

print("JSONL files for training and testing have been created.")