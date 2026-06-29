"""
Document loaders for various input formats.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from .pipeline import Document


def load_text_file(path: str, source_name: Optional[str] = None) -> Document:
    content = Path(path).read_text(encoding="utf-8")
    return Document(
        content=content, metadata={"source": source_name or Path(path).name, "type": "text"}
    )


def load_json_file(path: str, content_key: str = "content") -> List[Document]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        docs = []
        for item in data:
            text = item.get(content_key) or item.get("text") or json.dumps(item)
            docs.append(
                Document(
                    content=text,
                    metadata={
                        k: v for k, v in item.items() if k != content_key and isinstance(v, str)
                    },
                )
            )
        return docs
    else:
        return [Document(content=json.dumps(data, indent=2), metadata={"source": path})]


def load_markdown_file(path: str) -> Document:
    content = Path(path).read_text(encoding="utf-8")
    # Strip markdown syntax for cleaner embedding
    clean = re.sub(r"#{1,6}\s+", "", content)
    clean = re.sub(r"\*\*(.*?)\*\*", r"\1", clean)
    clean = re.sub(r"\*(.*?)\*", r"\1", clean)
    clean = re.sub(r"`{1,3}.*?`{1,3}", "", clean, flags=re.DOTALL)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    return Document(
        content=clean.strip(),
        metadata={"source": Path(path).name, "type": "markdown", "raw_path": path},
    )


def load_directory(directory: str, extensions: List[str] = None) -> List[Document]:
    """Recursively load all supported files from a directory."""
    if extensions is None:
        extensions = [".txt", ".md", ".json"]
    docs = []
    for path in Path(directory).rglob("*"):
        if path.suffix in extensions and path.is_file():
            try:
                if path.suffix == ".md":
                    docs.append(load_markdown_file(str(path)))
                elif path.suffix == ".json":
                    docs.extend(load_json_file(str(path)))
                else:
                    docs.append(load_text_file(str(path)))
            except Exception as e:
                print(f"[Loader] Skipped {path}: {e}")
    return docs


def load_from_strings(texts: List[str], source: str = "inline") -> List[Document]:
    """Create documents directly from strings (useful for testing)."""
    return [
        Document(content=t, metadata={"source": source, "index": i}) for i, t in enumerate(texts)
    ]
