import os
import json
import csv
import sys
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-#####"
client = OpenAI()

def load_test_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_data(filepath):
    data = {}
    with open(filepath, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            index = int(row['Sample Index'])
            data[index] = {
                'label': int(row['label']),
                'source_text': row['source_text'],
                'target_text': row['target_text'] or "This text does not contain detectable subjective bias."
            }
    return data

def extract_neutral_text(response):
    markers = ["The neutralized text is: ", "Neutralized text: ", "This is subjective bias text. "]
    for marker in markers:
        if marker in response:
            return response.split(marker, 1)[1].strip()
    return "No neutral text provided"


def evaluate_model(model, test_data, data, csv_file):
    correct = 0
    total = 0

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sample Index', 'Source Text', 'Predicted Neutral Text', 'True Neutral Text'])

        for index, item in enumerate(test_data):
            sample_index = index + 1
            messages = item["messages"]
            true_label = data[sample_index]['label']

            relevant_messages = [msg for msg in messages if msg['role'] != 'assistant']

            response = client.chat.completions.create(
                model=model,
                messages=relevant_messages,
                max_tokens=300
            )
            predicted_response = response.choices[0].message.content.strip()
            predicted_label = 1 if "This is subjective bias text" in predicted_response else 0
            predicted_neutral_text = extract_neutral_text(predicted_response) if predicted_label == 1 else "No bias detected"

            source_text = data.get(sample_index, {}).get('source_text', "Source text missing")
            true_neutral_text = data.get(sample_index, {}).get('target_text', "No target text provided")

            print(f"Sample {sample_index}:")
            print(f"Predicted Label: {predicted_label}, True Label: {true_label}")
            print(f"Predicted Response: {predicted_response}")
            print(f"True Response: {true_neutral_text}")
            print("Correct" if predicted_label == true_label else "Incorrect")
            print("-" * 80)

            if predicted_label == true_label:
                correct += 1
            total += 1

            if predicted_label == 1 and true_label == 1:
                csv_writer.writerow([sample_index, source_text, predicted_neutral_text, true_neutral_text])

        accuracy = correct / total if total > 0 else 0
        return accuracy


if __name__ == "__main__":
    base_dir = "models_evaluate"
    os.makedirs(base_dir, exist_ok=True)

    existing_models = [d for d in os.listdir(base_dir) if d.startswith('model_')]
    if existing_models:
        model_nums = [int(model.split('_')[-1]) for model in existing_models]
        model_num = max(model_nums) + 1
    else:
        model_num = 1

    folder_name = f'model_{model_num}'
    dataset_result_path = os.path.join(base_dir, folder_name)
    os.makedirs(dataset_result_path, exist_ok=True)

    output_folder_path = os.path.join(dataset_result_path, "model_output")
    os.makedirs(output_folder_path, exist_ok=True)

    results_file_path = os.path.join(output_folder_path, 'evaluation_output.txt')
    original_stdout = sys.stdout
    with open(results_file_path, 'w', encoding='utf-8') as f:
        sys.stdout = f

        pretrained_model_id = "gpt-3.5-turbo"
        finetuned_model_id = "ft:gpt-3.5-turbo-0125:personal::#####"
        test_data_path = "data/fine_tuning_test_data.jsonl"
        source_data_path = "data/test_data.csv"

        test_data = load_test_data(test_data_path)
        data = load_data(source_data_path)

        print("Evaluating pre-trained model...")
        pretrained_results_file = os.path.join(output_folder_path, 'pretrained_results.csv')
        pretrained_model_accuracy = evaluate_model(pretrained_model_id, test_data, data, pretrained_results_file)
        print(f"Pre-trained model accuracy: {pretrained_model_accuracy}\n")

        print("Evaluating fine-tuned model...")
        finetuned_results_file = os.path.join(output_folder_path, 'finetuned_results.csv')
        finetuned_model_accuracy = evaluate_model(finetuned_model_id, test_data, data, finetuned_results_file)
        print(f"Fine-tuned model accuracy: {finetuned_model_accuracy}\n")

        improvement = finetuned_model_accuracy - pretrained_model_accuracy
        print(f"Improvement: {improvement}")

    sys.stdout = original_stdout
    print(f"Evaluation complete, results saved to '{results_file_path}'")