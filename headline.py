from pathlib import Path

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

train_path = DATA_DIR / "LABELLED_TRAIN.csv"
dev_path = DATA_DIR / "LABELLED_DEV.csv"
test_path = DATA_DIR / "UNLABELLED_TEST.csv"

MODEL_PATH = BASE_DIR / "Best_headline_model"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)


# ============================================================
# DATASET
# ============================================================

class HeadlineDataset(Dataset):
    def __init__(self, articles, headlines, tokenizer, max_len=512):
        self.articles = articles
        self.headlines = headlines
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = "summarize: " + str(self.articles[idx])
        headline = str(self.headlines[idx])

        # Use the current Transformers tokenizer API.
        # encode_plus() is deprecated/incompatible with some newer
        # Transformers versions.
        inputs = self.tokenizer(
            article,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer(
            headline,
            max_length=50,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = targets["input_ids"].squeeze(0)

        # Ignore padding tokens when calculating T5 loss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "raw_headline": headline
        }


# ============================================================
# ROUGE
# ============================================================

def calculate_rouge_batch(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)

    return float(np.mean(scores))


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, tokenizer):
    model.train()
    total_loss = 0
    all_rouge_scores = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        raw_headlines = batch["raw_headline"]

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Generate headlines for ROUGE calculation.
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )

            predictions = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )

            rouge_score = calculate_rouge_batch(
                predictions,
                raw_headlines
            )
            all_rouge_scores.append(rouge_score)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "rouge-l": f"{rouge_score:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return (
        total_loss / len(dataloader),
        float(np.mean(all_rouge_scores))
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, dataloader, device, tokenizer):
    model.eval()
    total_loss = 0
    all_rouge_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            raw_headlines = batch["raw_headline"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )

            predictions = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )

            rouge_score = calculate_rouge_batch(
                predictions,
                raw_headlines
            )
            all_rouge_scores.append(rouge_score)

    return (
        total_loss / len(dataloader),
        float(np.mean(all_rouge_scores))
    )


# ============================================================
# TEST PREDICTIONS
# ============================================================

def generate_test_predictions(model, test_df, tokenizer, device):
    model.eval()

    test_articles = test_df["News Article"].fillna("").tolist()
    predictions = []

    with torch.no_grad():
        for article in tqdm(
            test_articles,
            desc="Generating Test Predictions"
        ):
            inputs = tokenizer(
                "summarize: " + str(article),
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )

            prediction = tokenizer.decode(
                generated[0],
                skip_special_tokens=True
            )

            predictions.append(prediction)

    test_df["Prediction"] = predictions

    output_path = BASE_DIR / "test_predictions.csv"
    test_df[["ID", "Prediction"]].to_csv(
        output_path,
        index=False
    )

    print(f"Predictions saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

print(
    "Initializing model and tokenizer..... "
    "Made by Divyanshu Kannaujiya"
)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train_df["News Article"].fillna("").tolist()
y_train = train_df["Caption"].fillna("").tolist()

X_dev = dev_df["News Article"].fillna("").tolist()
y_dev = dev_df["Caption"].fillna("").tolist()


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = HeadlineDataset(
    X_train,
    y_train,
    tokenizer
)

dev_dataset = HeadlineDataset(
    X_dev,
    y_dev,
    tokenizer
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=8
)


# ============================================================
# TRAINING SETUP
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model.to(device)

epochs = 1

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5
)

total_steps = len(train_loader) * epochs

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=1e-6
)


# ============================================================
# TRAINING LOOP
# ============================================================

best_rouge_l = -1.0
results = []

print("Starting training...")

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    train_loss, train_rouge = train_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        device,
        tokenizer
    )

    val_loss, val_rouge = evaluate(
        model,
        dev_loader,
        device,
        tokenizer
    )

    print(
        f"Train Loss: {train_loss:.4f}, "
        f"Train ROUGE-L: {train_rouge:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}, "
        f"Val ROUGE-L: {val_rouge:.4f}"
    )

    results.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_rouge": train_rouge,
        "val_loss": val_loss,
        "val_rouge": val_rouge
    })

    # Save the BEST model and tokenizer into THE SAME folder.
    if val_rouge > best_rouge_l:
        best_rouge_l = val_rouge

        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)

        print(
            f"Saved best model with ROUGE-L: "
            f"{val_rouge:.4f}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\nFinal Results:")

results_df = pd.DataFrame(results)
print(results_df)


# ============================================================
# TRAINING PLOTS
# ============================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(
    results_df["epoch"],
    results_df["train_loss"],
    label="Train Loss"
)
plt.plot(
    results_df["epoch"],
    results_df["val_loss"],
    label="Val Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(
    results_df["epoch"],
    results_df["train_rouge"],
    label="Train ROUGE-L"
)
plt.plot(
    results_df["epoch"],
    results_df["val_rouge"],
    label="Val ROUGE-L"
)
plt.xlabel("Epoch")
plt.ylabel("ROUGE-L Score")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# TEST PREDICTIONS
# ============================================================

print(
    "Generating predictions for test data......"
    "Made by Divyanshu Kannaujiya"
)

generate_test_predictions(
    model,
    test_df,
    tokenizer,
    device
)
