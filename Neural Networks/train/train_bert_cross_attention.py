import os
import json
import time
import random
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from models.dataset import (
    read_jsonl,
    YelpBertDataset,
    bert_collate_fn,
    StandardScalerFromList,
    get_meta_feature_vector,
)
from models.bert_cross_attention import BERTCrossAttentionClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes(task: str) -> int:
    if task == "sentiment":
        return 3
    elif task == "rating":
        return 9
    else:
        raise ValueError(f"Unsupported task: {task}")


def build_scaler(train_path: str) -> StandardScalerFromList:
    train_samples = read_jsonl(train_path)
    train_meta = [get_meta_feature_vector(x) for x in train_samples]
    scaler = StandardScalerFromList()
    scaler.fit(train_meta)
    return scaler


def evaluate(model, dataloader, criterion, device) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            meta_features = batch["meta_features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                meta_features=meta_features,
            )
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(all_labels))
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, grad_clip: float = 1.0) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_features = batch["meta_features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_features=meta_features,
        )
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(all_labels))
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT+CrossAttention on Yelp unified dataset.")
    parser.add_argument("--task", type=str, default="sentiment", choices=["sentiment", "rating"])
    parser.add_argument("--train_path", type=str, default="data/splits/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/splits/val.jsonl")
    parser.add_argument("--test_path", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/bert_cross_attention")

    parser.add_argument("--pretrained_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_meta_tokens", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze_bert", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    scaler = build_scaler(args.train_path)

    with open(os.path.join(args.output_dir, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler.state_dict(), f, ensure_ascii=False, indent=2)

    train_dataset = YelpBertDataset(
        data_path=args.train_path,
        task=args.task,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_field=args.text_field,
        use_meta_features=True,
        scaler=scaler,
    )
    val_dataset = YelpBertDataset(
        data_path=args.val_path,
        task=args.task,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_field=args.text_field,
        use_meta_features=True,
        scaler=scaler,
    )
    test_dataset = YelpBertDataset(
        data_path=args.test_path,
        task=args.task,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_field=args.text_field,
        use_meta_features=True,
        scaler=scaler,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: bert_collate_fn(batch, pad_token_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: bert_collate_fn(batch, pad_token_id=tokenizer.pad_token_id),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: bert_collate_fn(batch, pad_token_id=tokenizer.pad_token_id),
    )

    meta_dim = train_dataset[0]["meta_features"].shape[0]
    model = BERTCrossAttentionClassifier(
        pretrained_model_name=args.pretrained_model_name,
        num_classes=get_num_classes(args.task),
        meta_dim=meta_dim,
        num_meta_tokens=args.num_meta_tokens,
        dropout=args.dropout,
        num_heads=args.num_heads,
        freeze_bert=args.freeze_bert,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_val_f1 = -1.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        elapsed = time.time() - start_time

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_sec": elapsed,
        }
        history.append(record)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"time={elapsed:.2f}s"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to: {best_model_path}")

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("[INFO] Loading best model for final test...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n========== Final Test Results ==========")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    report = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        digits=4,
    )
    print(report)

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "classification_report": report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()