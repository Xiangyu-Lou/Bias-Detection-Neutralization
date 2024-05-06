# Bias-Detection-Neutralization 🧠

This project focuses on fine-tuning GPT-3.5 to detect and neutralize biases in text, utilizing the Wiki Neutrality Corpus.

## Table of Contents 📖

| [Dataset Information](#dataset-information) | [Setup and Usage](#setup-and-usage) | [Model Evaluation](#model-evaluation) | [Evaluation Results](#evaluation-results) | [Visualizations](#visualizations) |
|---------------------------------------------|-------------------------------------|---------------------------------------|------------------------------------------|----------------------------------|

## Dataset Information 📚

- **Dataset Name**: Wiki Neutrality Corpus
- **Authors**: Reid Pryzant, Richard Diehl Martinez, Nathan Dass, Sadao Kurohashi, Dan Jurafsky, Diyi Yang
- **Source**: [Kaggle](https://www.kaggle.com/datasets/chandiragunatilleke/wiki-neutrality-corpus)
- **License**: Please refer to the [Kaggle page](https://www.kaggle.com/datasets/chandiragunatilleke/wiki-neutrality-corpus) for license details.

## Setup and Usage 🔧

### Prerequisites

- Ensure you have a Kaggle account to download the dataset.
- An OpenAI API key is required for model training.

### Installation Steps

1. **Dataset Preparation**
   - Download the `biased.full` file from the Kaggle link provided above.
   - Place the downloaded file in the `data` folder of your project directory.

2. **Configuration**
   - Open `training.ipynb`.
   - Replace `sk-#####` with your actual OpenAI API key in the notebook.

3. **Data Processing**
   - Execute the `Split the dataset Cell` in `training.ipynb` to divide the dataset into training and validation sets.
   - Run the `Upload dataset Cell` to upload the dataset to OpenAI and retrieve the file ID.

4. **Model Training**
   - Replace `file-#####` with your obtained file ID.
   - Execute the `Fine-tuning the model Cell` to start the fine-tuning process.
  
5. **Evaluation**
   - Download the pre-trained `word2vec` model form [google news word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and place it in the root directory.
   - Run all cells in `evaluation.ipynb` to evaluate the fine-tuned model.

## Evaluation Results 📊

The results below summarize the performance of the fine-tuned versus the pre-trained models:

| **Model**          | **Bias Detection Accuracy** | **BLEU Score** | **WMD** | **Accuracy of Bias Removal** |
|--------------------|-----------------------------|----------------|---------|------------------------------|
| Fine-tuned model   | 69.0%                       | 0.85           | 0.06    | 80.2%                        |
| Pre-trained model  | 59.4%                       | 0.29           | 0.29    | 41.3%                        |

## Visualizations 🖼️

Here are some visual insights into the model's performance:

![BLEU Score Distribution](/pictures/bleu_kde.png)
![Word Mover's Distance (WMD) Distribution](/pictures/wmd_kde.png)
