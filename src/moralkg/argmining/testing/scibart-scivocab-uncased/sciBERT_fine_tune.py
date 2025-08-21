# sciBERT_fine_tune.py
#!/usr/bin/env python3


import argparse
from pie_datasets import load_dataset # type: ignore
from pie_datasets.builders.brat import BratDocumentWithMergedSpans # type: ignore
from pytorch_ie.documents import TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions  # type: ignore
from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset # type: ignore


def flatten_spans(doc):
    """
    Convert one PIE document into a list of {text,label} examples.
    """
    out = []
    for span in doc.labeled_spans:
        txt = doc.text[span.start : span.end]
        if span.label in ("MajorClaim", "Claim"):
            y = 0
        elif span.label == "Premise":
            y = 1
        else:
            y = 2
        out.append({"text": txt, "label": y})
    return out

def main():
    p = argparse.ArgumentParser(description="Fine-tune SciBERT on PIE AAE2 spans")
    p.add_argument("--model_name",    default="allenai/scibert_scivocab_uncased")
    p.add_argument("--output_dir",    default="scibert-argmine")
    p.add_argument("--epochs",   type=int,   default=3)
    p.add_argument("--train_bs", type=int,   default=8)
    p.add_argument("--eval_bs",  type=int,   default=None)
    p.add_argument("--lr",       type=float, default=2e-5)
    args = p.parse_args()

    # 1) Load & convert PIE AAE2
    raw = load_dataset("pie/aae2")
    ds  = raw.to_document_type(TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions)

    # 2) Build train/val tables
    train_rows = sum((flatten_spans(d) for d in ds["train"]), [])
    val_rows   = sum((flatten_spans(d) for d in ds["test" ]), [])
    hf_train = Dataset.from_list(train_rows)
    hf_val   = Dataset.from_list(val_rows)

    # 3) Tokenizer & tokenization fn
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize_batch(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)
    tokenized_train = hf_train.map(tokenize_batch, batched=True)
    tokenized_val   = hf_val.map(tokenize_batch, batched=True)

    # 4) Model + label mapping
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3
    )
    label_names = ["claim", "premise", "non-argument"]
    model.config.id2label = {i: lab for i, lab in enumerate(label_names)}
    model.config.label2id = {lab: i for i, lab in enumerate(label_names)}

    # 5) Trainer setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs or args.train_bs,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # 6) Train & save
    trainer.train()
    trainer.save_model(f"{args.output_dir}/best_model")
    print(f"Model and tokenizer saved to {args.output_dir}/best_model")

if __name__ == "__main__":
    main()