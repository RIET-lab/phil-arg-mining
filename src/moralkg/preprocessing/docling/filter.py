from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


_LONE_ACCENT = {
    "a´": "á",
    "e´": "é",
    "i´": "í",
    "o´": "ó",
    "u´": "ú",
    "a¨": "ä",
    "e¨": "ë",
    "i¨": "ï",
    "o¨": "ö",
    "u¨": "ü",
    "a^": "â",
    "e^": "ê",
    "i^": "î",
    "o^": "ô",
    "u^": "û",
    "a`": "à",
    "e`": "è",
    "i`": "ì",
    "o`": "ò",
    "u`": "ù",
    "a~": "ã",
    "o~": "õ",
    "n~": "ñ",
}
_LONE_ACCENT.update({k.upper(): v.upper() for k, v in _LONE_ACCENT.items()})


def _ditch_combining_diacritics(text: str) -> str:
    for orig, repl in _LONE_ACCENT.items():
        text = text.replace(orig, repl)
    text = re.sub(r"[\u0300-\u036F]", "", text)
    return re.sub(r"(?:\xa8|[\u02C0-\u02DF])", "", text)


def _is_letter(x: str) -> bool:
    return x in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _is_date(x: str) -> bool:
    return re.match(r".*([1-3][0-9]{3})", x) is not None


def _header_footer_filter(para: str) -> str:
    try:
        if len(para) < 50:
            s = para.strip()
            if not s:
                return para
            if s[0] == "©":
                return ""
            if s[0] == "r":
                parts = s.split(" ")
                if len(parts) >= 2 and _is_date(parts[1]):
                    return ""
            if s.split(" ")[0] == "copyright":
                parts = s.split(" ")
                if len(parts) > 1 and _is_date(parts[1]):
                    return ""
    except Exception:
        return para
    return para


def _replace_hyphenated(text: str) -> str:
    text = re.sub(r"-[?\s]\n{1,2}(\w+ *)", r"\1\n", text)
    return re.sub(r"-\s{1,2}(\w+ *)", r"\1", text)


def _remove_leading_and_trailing_nums(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(\d+)", "", text)
    text = re.sub(r"(\d+)$", "", text)
    return text.strip()


def _cid_percentage(text: str) -> float:
    n_matches = len(re.findall("\(cid:[0-9]+\)", text))
    return ((n_matches * 8) / len(text)) if text else 0.0


def _remove_cid(text: str) -> str:
    return re.sub("\(cid:[0-9]+\)", "", text)


def _filter_double_whitespace(text: str) -> str:
    return re.sub("\s\s+", " ", text)


def _filter_newlines(text: str) -> str:
    return re.sub("\n", " ", text)


_UNICODE_MAPPING = {
    r"(\B)\u00DF": r"\1ss",
    "\xa0": " ",
    r"[\u2018\u2019]": r"'",
    r"[\u201C\u201D]": r'"',
    r"[\xad\u2014]": r"-",
    r"\xb7": r"*",
}


def fix_unicode(txt: str) -> str:
    for search, replace in _UNICODE_MAPPING.items():
        txt = re.subn(search, replace, txt)[0]
    return unicodedata.normalize("NFKC", txt)


def filter_text(text: str) -> str:
    text = _replace_hyphenated(text)
    paragraphs = text.split("\n\n")
    cleaned: List[str] = []
    for para in paragraphs:
        para = _filter_newlines(para)
        para = _header_footer_filter(para)
        para = _filter_double_whitespace(para)

        non_empty_line_lengths = [len(line) for line in para.split("\n") if line]
        mean_line_len = (sum(non_empty_line_lengths) / len(non_empty_line_lengths)) if non_empty_line_lengths else 0
        if mean_line_len < 2.0:
            continue

        if _cid_percentage(para) > 0.1:
            continue

        letterness = sum(1 for ch in para if _is_letter(ch)) / len(para) if para else 0
        if letterness < 0.40:
            continue

        para = _ditch_combining_diacritics(
            fix_unicode(_remove_cid(_remove_leading_and_trailing_nums(para)))
        )
        if para:
            cleaned.append(para)

    return "\n\n".join(cleaned)


@dataclass
class DoclingTextFilter:
    overwrite: bool = False

    def process_file(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        text = input_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = filter_text(text)
        if self.overwrite and output_path is None:
            input_path.write_text(cleaned, encoding="utf-8")
            return input_path
        out = output_path or input_path.with_suffix(".filtered" + input_path.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(cleaned, encoding="utf-8")
        return out

    def process_directory(self, input_dir: Path, output_dir: Optional[Path] = None, exts: Optional[Iterable[str]] = None) -> int:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(str(input_dir))
        exts = tuple((ext if ext.startswith(".") else f".{ext}") for ext in (exts or [".md", ".txt"]))
        count = 0
        for path in input_dir.glob("**/*"):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            out = None
            if output_dir is not None and not self.overwrite:
                out_rel = path.relative_to(input_dir)
                out = Path(output_dir) / out_rel
            self.process_file(path, out)
            count += 1
        return count


__all__ = ["fix_unicode", "filter_text", "DoclingTextFilter"]


