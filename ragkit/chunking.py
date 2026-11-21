"""Two chunking strategies: one chunk per section, or fixed-size word windows with overlap."""
from __future__ import annotations

from dataclasses import dataclass

from .loader import Section


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    title: str
    sections: tuple[str, ...]   # sections this chunk covers (a window can cross a boundary)
    text: str                   # what gets embedded (includes the manual title)
    body: str = ""              # the manual text alone, used in answers

    @property
    def citation(self) -> str:
        return f"{self.doc}#{self.sections[0]}"


def chunk_by_section(sections: list[Section]) -> list[Chunk]:
    # The manual title is added so the product name is searchable in every chunk.
    return [Chunk(f"{s.doc}-{i}", s.doc, s.title, (s.section,), f"{s.title}. {s.section}. {s.text}", s.text)
            for i, s in enumerate(sections)]


def chunk_by_window(sections: list[Section], size: int = 40, overlap: int = 10) -> list[Chunk]:
    """Slide a window of `size` words over each manual, moving `size - overlap` words each time."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    chunks: list[Chunk] = []
    by_doc: dict[str, list[Section]] = {}
    for s in sections:
        by_doc.setdefault(s.doc, []).append(s)
    for doc, secs in by_doc.items():
        words: list[tuple[str, str]] = []           # (word, section it came from)
        for s in secs:
            words += [(w, s.section) for w in s.text.split()]
        step = size - overlap
        for n, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
            part = words[start:start + size]
            covered = tuple(dict.fromkeys(sec for _, sec in part))
            body = " ".join(w for w, _ in part)
            chunks.append(Chunk(f"{doc}-w{n}", doc, secs[0].title, covered, f"{secs[0].title}. {body}", body))
    return chunks
