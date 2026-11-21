"""Read the markdown manuals into sections."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Section:
    doc: str        # file name without .md, for example "switcher"
    title: str      # manual title from the first heading
    section: str    # section heading
    text: str


def load_manuals(folder: str | Path) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(Path(folder).glob("*.md")):
        title, current, buf = path.stem, None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                if current and buf:
                    sections.append(Section(path.stem, title, current, " ".join(buf).strip()))
                current, buf = line[3:].strip(), []
            elif current and line.strip():
                buf.append(line.strip())
        if current and buf:
            sections.append(Section(path.stem, title, current, " ".join(buf).strip()))
    if not sections:
        raise ValueError(f"no sections found in {folder}")
    return sections
