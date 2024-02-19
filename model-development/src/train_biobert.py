import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(args):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        accuracy = accuracy_score(labels, preds)
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    dataset = load_dataset(args.dataset_path)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "dmis-lab/biobert-v1.1", num_labels=8
    )

    training_args = TrainingArguments(
        output_dir="models/biobert",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="no",
        do_eval=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.model.save_pretrained("models/biobert")
    trainer.tokenizer.save_pretrained("models/biobert/tokenizer")
