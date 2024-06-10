import os
import numpy as np
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import f1_score

import pandas as pd
from sklearn.model_selection import train_test_split

# JSONLファイルの読み込み
file_path = "oscar_mixtral_scored.jsonl"
df = pd.read_json(file_path, lines=True)

# dfからscoreが5より大きいレコードを削除
df = df[df['score'] <= 5]

# データをtrain, tempに分割
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['score'], random_state=42)

# tempをval, testに分割
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['score'], random_state=42)

# 分割されたデータフレームのサイズを表示
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

# Datasetとして読み込み
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained("pkshatech/GLuCoSE-base-ja")

def preprocess(examples):
    batch = tokenizer(examples["text"], truncation=True)
    batch["labels"] = examples["score"]
    return batch

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("pkshatech/GLuCoSE-base-ja", num_labels=6, classifier_dropout=0.0, hidden_dropout_prob=0.0)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, predictions, average="macro")}

training_args = TrainingArguments(
    output_dir="outputs",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=1,
    learning_rate=1e-5,
    num_train_epochs=20,
    seed=0,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
