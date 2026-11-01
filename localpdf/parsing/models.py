from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageText:
    page: int  # 1-based
    text: str
    ocr: bool = False


@dataclass
class ExtractedDoc:
    pages: list[PageText]
    n_pages: int
    ocr_pages: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
