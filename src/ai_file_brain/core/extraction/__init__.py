from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from ai_file_brain.core.extraction.docx import DocxExtractor
from ai_file_brain.core.extraction.image import ImageExtractor
from ai_file_brain.core.extraction.pdf import PdfExtractor
from ai_file_brain.core.extraction.plain_text import PlainTextExtractor
from ai_file_brain.core.models import ExtractionResult


class UnsupportedFileTypeError(ValueError):
    pass


@runtime_checkable
class TextExtractor(Protocol):
    async def extract(self, file_path: str) -> ExtractionResult: ...


_image_extractor = ImageExtractor()
_plain_text_extractor = PlainTextExtractor()

# Plain-text formats and source-code-like files all share PlainTextExtractor —
# they're just UTF-8 text. Listed explicitly so users know what gets indexed.
_PLAIN_TEXT_EXTENSIONS: tuple[str, ...] = (
    ".txt",
    ".md",
    ".rst",
    # source code
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
    ".sh",
    ".bash",
    ".ps1",
    ".sql",
    # config / data
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".ini",
    ".cfg",
    ".env",
)

_EXTRACTORS: dict[str, TextExtractor] = {
    ".pdf": PdfExtractor(),
    ".docx": DocxExtractor(),
    ".png": _image_extractor,
    ".jpg": _image_extractor,
    ".jpeg": _image_extractor,
    ".tiff": _image_extractor,
    ".tif": _image_extractor,
    ".bmp": _image_extractor,
    ".webp": _image_extractor,
}
for _ext in _PLAIN_TEXT_EXTENSIONS:
    _EXTRACTORS[_ext] = _plain_text_extractor

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXTRACTORS)


def get_extractor(file_path: str) -> TextExtractor:
    ext = os.path.splitext(file_path)[1].lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise UnsupportedFileTypeError(f"No extractor for extension '{ext}' (path={file_path!r})")
    return extractor


def is_supported(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in _EXTRACTORS


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ExtractionResult",
    "TextExtractor",
    "UnsupportedFileTypeError",
    "get_extractor",
    "is_supported",
]
