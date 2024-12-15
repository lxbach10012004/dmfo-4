# Data Mining Field-oriented Final Term Project

This repository contains the implementation for the final term project of the course **"Data Mining Field-oriented"**. The task involves **multilabel classification of news articles**, and we have explored three methods to solve this problem: 
1. **BERT_CRF**
2. **LLM (Large Language Models)**
3. **RULE-based method**

---

## Repository Structure

### 1. **BERT_CRF**
This folder contains our implementation of a BERT-based model with a Conditional Random Field (CRF) layer on top, fine-tuned for Named Entity Recognition (NER) tasks. The implementation includes the following files:

- **`data_prep.ipynb`**:
  - Converts raw datasets to an NER-compatible dataset following the B-I-O tagging scheme.
  - Splits long articles into smaller samples to fit the model's input size.
  - **Inputs**: `articles_training.tsv`, `articles_testing.tsv`
  - **Outputs**: Transformed NER datasets: `split_train_dataset.csv`, `split_test_dataset.csv`, and optionally `split_private_test_dataset.csv`.

- **`train.ipynb`**:
  - Defines the BERT+CRF model architecture and handles the training process.
  - **Inputs**: Transformed and split NER datasets from `data_prep.ipynb`.
  - **Outputs**: Best model checkpoint (`ner_lm_crf_checkpoint.pt`) and predictions for the test set.

- **`infer.ipynb`**:
  - Loads a trained model checkpoint and makes predictions on a given dataset.
  - **Outputs**: Predicted dataset.

### 2. **LLM**
This folder contains code to leverage APIs from three popular large language models: **GPT**, **Gemini**, and **Llama**. The implementation focuses on:

- Calling these models using their respective APIs.
- Our engineered prompts designed to optimize predictions for the multilabel classification task.

### 3. **RULE**
This folder implements a simple rule-based approach for multilabel classification:

- **`rule.ipynb`**:
  - Extracts all unique tags from the training set and searches for these tags in the content of the test set to make predictions.
  - Includes a post-processing step to remove partial or invalid word predictions.

---

## How to Use

### 1. **BERT_CRF**
1. Run `data_prep.ipynb` to preprocess the raw dataset and generate transformed NER datasets.
2. Use `train.ipynb` to train the model and save the best checkpoint.
3. Run `infer.ipynb` with the trained model checkpoint to generate predictions.

### 2. **LLM**
1. Modify the API credentials and settings in the code as needed.
2. Follow the instructions to call GPT, Gemini, or Llama with the provided prompts.
3. Evaluate predictions using the respective outputs.

### 3. **RULE**
1. Run `rule.ipynb` to predict tags for the test set based on a simple tag-matching rule.
2. The notebook includes steps for both prediction and post-processing.
