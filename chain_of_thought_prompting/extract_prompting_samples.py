import pandas as pd

def extract_prompting_samples(model_number, num_samples):
    '''
    Extracts the smallest BLEU Scores from the finetuned results of a model and saves them to a csv file.
    
    Parameters:
    model_number (int): The model number to extract the samples from.
    num_samples (int): The number of samples to extract.
    
    Returns:
    smallest_bleu_scores (DataFrame): The DataFrame containing the smallest BLEU Scores.
    '''
    
    finetuned_result = f"./models_evaluate/model_{model_number}/evaluate_result/bleu_score_finetuned.csv"
    df = pd.read_csv(finetuned_result)

    # Get the smallest BLEU Scores and save them to a csv file
    smallest_bleu_scores = df.nsmallest(num_samples, 'BLEU Score')
    smallest_bleu_scores.to_csv(f'./models_evaluate/model_{model_number}/prompting/samples_prompting.csv', index=False)
    
    return smallest_bleu_scores
    
if __name__ == '__main__':
    samples = extract_prompting_samples(4, 10)
    print(samples)
