"""
End-to-end NLP Project:
Sentiment Analysis on IMDB Movie Reviews using NLTK + Hugging Face Transformers

What this script does:
1. Loads a movie review dataset (IMDB) using 🤗 Datasets.
2. Uses NLTK for basic text preprocessing (lowercasing, tokenization, stopword removal, lemmatization).
3. Uses a pretrained BERT-like model (DistilBERT) from Hugging Face.
4. Fine-tunes the model on a small subset of IMDB to classify reviews as Positive / Negative.
5. Evaluates the model and shows example predictions.

Requirements (run these in terminal / Colab first):

    pip install nltk datasets transformers scikit-learn

If you're running NLTK for the first time, it will download some resources.
"""

# 1. IMPORT LIBRARIES

import os
import random

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import accuracy_score, f1_score, classification_report


# 2. NLTK SETUP
# =========================

# Download necessary NLTK resources (will only download once)
nltk.download("punkt")        # For tokenization
nltk.download("stopwords")    # For stopword removal
nltk.download("wordnet")      # For lemmatization

# Initialize global objects for preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()



# 3. TEXT PREPROCESSING FUNCTION
# =========================

def clean_text(text: str) -> str:
    """
    Basic text preprocessing using NLTK.

    Steps:
    1. Lowercase the text.
    2. Tokenize into words.
    3. Remove punctuation tokens and non-alphabetic tokens.
    4. Remove stopwords (e.g. 'the', 'is', 'and').
    5. Lemmatize each token to its base form (e.g. 'running' -> 'run').
    6. Join back into a cleaned string.

    Note:
    - This is a demo pipeline. In production, you might tune it carefully
      or even skip heavy cleaning because transformers handle raw text well.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Tokenize into words
    tokens = nltk.word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        # 3. Keep only alphabetic tokens (remove punctuation, numbers, etc.)
        if not token.isalpha():
            continue

        # 4. Remove stopwords
        if token in stop_words:
            continue

        # 5. Lemmatize token (default POS = noun; good enough for demo)
        lemma = lemmatizer.lemmatize(token)

        cleaned_tokens.append(lemma)

    # 6. Join tokens back into a single string
    return " ".join(cleaned_tokens)



# 4. LOAD DATASET (IMDB)
# =========================

"""
We use the IMDB movie reviews dataset from Hugging Face Datasets.

- Each example has:
    "text": the movie review (string)
    "label": 0 (negative) or 1 (positive)

We will:
- Use a smaller subset for fast training (e.g. 2000 train, 500 test).
- In a real project, you would train on the full dataset.
"""

print("Loading IMDB dataset...")
raw_datasets = load_dataset("imdb")

# Use a small subset for demonstration (otherwise training takes long)
train_size = 2000
test_size = 500

small_train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(train_size))
small_test_dataset = raw_datasets["test"].shuffle(seed=42).select(range(test_size))

print(f"Train examples: {len(small_train_dataset)}")
print(f"Test examples: {len(small_test_dataset)}")



# 5. LOAD TOKENIZER & MODEL (DISTILBERT)
# =========================

"""
We use a DistilBERT model that is already pre-trained and
fine-tuned on sentiment data: 'distilbert-base-uncased'.

We will:
- Use the tokenizer to convert text into model inputs.
- Fine-tune the model on IMDB subset.
"""

model_name = "distilbert-base-uncased"

print(f"Loading tokenizer and model: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification: 0=negative, 1=positive
)


# 6. TOKENIZATION + PREPROCESSING PIPELINE
# =========================

def preprocess_function(examples):
    """
    This function will be applied to each batch of examples from the dataset.

    Steps:
    1. Clean the raw text via NLTK (clean_text).
    2. Use Hugging Face tokenizer to convert cleaned text into:
       - input_ids: token IDs
       - attention_mask: which tokens are real vs padding
    3. Return a dict with those fields so the Trainer can use them.

    Note:
    - We use truncation=True to cut long reviews (max_length default is model-specific).
    - You can specify max_length=256 or 512 if needed.
    """
    # Apply our NLTK-based cleaning to each review
    cleaned_texts = [clean_text(t) for t in examples["text"]]

    # Tokenize the cleaned text
    return tokenizer(
        cleaned_texts,
        truncation=True,
        padding=False,   # We'll pad later with DataCollator
    )


print("Preprocessing datasets (this can take a bit)...")
tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# Set the format for PyTorch
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

tokenized_train.set_format("torch")
tokenized_test.set_format("torch")



# 7. DATA COLLATOR (DYNAMIC PADDING)
# =========================

"""
DataCollatorWithPadding:
- Dynamically pads batches to the longest sequence in the batch.
- Saves memory vs. padding everything to max_length.
"""

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# 8. METRICS FOR EVALUATION
# =========================

def compute_metrics(eval_pred):
    """
    Compute accuracy and F1 score for model evaluation.

    eval_pred:
    - A tuple (logits, labels) from the Trainer.

    Returns:
    - A dict with metric names and values.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": acc, "f1": f1}



# 9. TRAINING CONFIGURATION (TrainingArguments)
# =========================

"""
TrainingArguments control how the Trainer will fine-tune the model.

Key parameters:
- output_dir: where to save checkpoints.
- evaluation_strategy: when to run eval (e.g. "epoch" or "steps").
- per_device_train_batch_size: batch size per GPU/CPU.
- num_train_epochs: how many passes over the training data.
- logging_steps: how often to log training progress.
- save_strategy: when to save checkpoints.
"""

training_args = TrainingArguments(
    output_dir="./imdb-sentiment-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,      # For demo; increase to 3-5 for better results
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
)



# 10. CREATE TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



# 11. TRAIN THE MODEL
# =========================

print("Starting training...")
trainer.train()
print("Training complete!")



# 12. EVALUATE THE MODEL
# =========================

print("Evaluating model on test set...")
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# More detailed report (optional)
# Get predictions for the test set
predictions_output = trainer.predict(tokenized_test)
logits = predictions_output.predictions
labels = predictions_output.label_ids
preds = np.argmax(logits, axis=-1)

print("\nClassification report on test subset:")
print(classification_report(labels, preds, target_names=["negative", "positive"]))



# 13. INFERENCE / PREDICTION ON NEW TEXT
# =========================

def predict_sentiment(review_text: str) -> None:
    """
    Run the fine-tuned model on a custom review string
    and print the predicted sentiment with probability.
    """
    model.eval()

    # 1. Clean the input text with NLTK pipeline
    cleaned = clean_text(review_text)

    # 2. Tokenize
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    # 3. Move to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # 4. Get model outputs (logits)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_label = int(np.argmax(probs))

    label_name = "positive" if pred_label == 1 else "negative"

    print(f"Original review: {review_text}")
    print(f"Cleaned review : {cleaned}")
    print(f"Predicted label: {label_name}")
    print(f"Probabilities  : negative={probs[0]:.3f}, positive={probs[1]:.3f}")
    print("-" * 60)


# Test with a few custom examples
print("\n=== Example Predictions ===")
predict_sentiment("I absolutely loved this movie, it was fantastic!")
predict_sentiment("This was the worst film I have ever seen. Boring and slow.")
predict_sentiment("It was okay, not great but not terrible either.")
