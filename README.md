# Bias-Detection-Neutralization
Fine-tuning GPT3.5 for bias detection and neutralization modifications

## Dataset Information

- **Dataset Name**: Wiki Neutrality Corpus
- **Author**: Reid Pryzant, Richard Diehl Martinez, Nathan Dass, Sadao Kurohashi, Dan Jurafsky, Diyi Yang
- **Source**: Kaggle
- **Download Link**: [Wiki Neutrality Corpus on Kaggle](https://www.kaggle.com/datasets/chandiragunatilleke/wiki-neutrality-corpus)
- **License**: Please see the Kaggle page for license details.

## How to Use:
- Download the dataset (`biased.full`) and place it in the `data` folder.
- Run `data_prepare.py`
- Prepare your own OpenAI API key and replace `sk-#####` in `fine_tuning.py` and `model_evaluate.py` with your key.
- Run `fine_tuning.py` to fine-tune your own model. You need to uncomment the first section of the code to upload the file. After obtaining the file ID, 
replace `file-#####` with it. Then, run the second section of the code for fine-tuning. These two steps require you to alternately comment out sections of the code.
- Run `model_evaluate.py` to evaluate the model's performance. You need to replace `ft:gpt-3.5-turbo-0125:personal::#####` with your own fine-tuned model ID. The result will be saved in `models_evaluate/model_n`.