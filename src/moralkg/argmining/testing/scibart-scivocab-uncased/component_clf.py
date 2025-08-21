# component_clf.py
#!/usr/bin/env python3

import argparse, json, warnings
warnings.filterwarnings("ignore", module="transformers.utils.generic")

from transformers import ( # type: ignore
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from load_texts import load_texts       # type: ignore
from sentence_splitter import split_sentences_spacy  # type: ignore
import torch # type: ignore

def classify_components(model, tokenizer, sentences, batch_size):
    """
    Returns list of (sent, label_id, score) for a batch of sentences.
    """
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(batch, 
                        padding=True, 
                        truncation=True, 
                        max_length=256, 
                        return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1).tolist()
        for sent, pid, p in zip(batch, preds, probs):
            score = p[pid].item()
            results.append((sent, pid, score))
    return results


def main():
    p = argparse.ArgumentParser(description="Classify components in papers")
    p.add_argument("--num_papers", type=int, default=1,
                   help="How many documents to process; -1 = all")
    p.add_argument("--max_sentences", type=int, default=-1,
                   help="Max sentences per doc; -1 = all")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--checkpoint", default="scibert-argmine/best_model")
    p.add_argument("--output", default="results/predictions.jsonl")
    args = p.parse_args()

    # Load tokenizer + model
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    cfg = AutoConfig.from_pretrained(args.checkpoint)
    if hasattr(cfg, "id2label"):
        cfg.id2label = {int(k):v for k,v in cfg.id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, config=cfg)
    model.eval()

    # Load texts
    texts = load_texts(max_files=(None if args.num_papers<0 else args.num_papers))
    docs_sents = split_sentences_spacy(texts)

    # Classify & collect
    all_records = []
    for doc_id, sents in enumerate(docs_sents):
        sents = sents if args.max_sentences<0 else sents[:args.max_sentences]
        preds = classify_components(model, tok, sents, args.batch_size)
        for sent_id, (sent, pid, score) in enumerate(preds):
            all_records.append({
                "doc_id":   doc_id,
                "sent_id":  sent_id,
                "sentence": sent,
                "label":    cfg.id2label[pid],
                "score":    score
            })

    # Save to JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_records)} predictions to {args.output}")

if __name__ == "__main__":
    main()