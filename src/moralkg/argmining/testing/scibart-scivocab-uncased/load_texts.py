# load_texts.py


from pathlib import Path
import re
from typing import List, Union

_DATA_DIR = Path("../../afk21005/philpapers/The-Pile-PhilPapers/data/text/")
# NOTE: .txt files also stored in data/docling_batch/ - that directory also 
# contains: .md, .html, .json, .yaml, .doctags.txt


def normalize_text(text: str) -> str:
    """
    Quickly normalize text by collapsing internal newlines and excess spaces.
    Ignores paragraph breaks.
    """
    paras = re.split(r'\n{2,}', text)
    cleaned = [' '.join(p.split()) for p in paras]
    return '\n\n'.join(cleaned)


def load_texts(folder: Union[str, Path] = _DATA_DIR, max_files: int = None) -> List[str]:
    """
    Load up to max_files .txt files from folder, returning their normalized contents.
    """
    folder = Path(folder)
    print(f"[load_texts] Loading up to {max_files or 'ALL'} .txt files from {folder}")
    
    texts = []
    for fp in folder.glob("*.txt"):
        if max_files and len(texts) >= max_files:
            break
        texts.append(normalize_text(fp.read_text(encoding="utf-8")))
    print(f"[load_texts] Loaded {len(texts)} documents\n")
    
    return texts

# Test the function
if __name__ == "__main__":
    texts = load_texts(max_files=1)
    print(f"Sample (first 200 chars):\n{texts[0][:200]}…")    
    