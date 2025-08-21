# sentence_splitter.py

import spacy # type: ignore
from typing import List

nlp = spacy.load(
    "en_core_web_sm", 
    disable=["tagger", "parser", "ner", "lemmatizer"]
    )
nlp.add_pipe("sentencizer", first=True)
nlp.max_length = 10_000_000


def split_sentences_spacy(docs: List[str]) -> List[List[str]]:
    """
    Split each document into a list of sentences.
    Returns a list of sentence‐lists, one per doc.
    """
    print(f"[split_sentences] Splitting {len(docs)} documents into sentences…")
    result = []
    for i, doc in enumerate(nlp.pipe(docs, batch_size=8), start=1):
        sents = [s.text.strip() for s in doc.sents]
        result.append(sents)
        print(f"[split_sentences]  Doc {i}/{len(docs)} → {len(sents)} sentences")
    print()
    return result

# Test the function
if __name__ == "__main__":
    from load_texts import load_texts
    limit = 5
    sentences = split_sentences_spacy(load_texts(max_files=limit))
    print(f"{limit} papers processed.")
    
    if sentences and sentences[0]:
        print("\nFirst 10 sentences of the first processed paper:")
        for sentence in sentences[0][:10]:
            print(sentence)
            