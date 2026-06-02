from __future__ import annotations

import asyncio
import logging
import struct

from ai_file_brain.core.models import ExtractionResult

logger = logging.getLogger(__name__)

# The binary PowerPoint (.ppt) format stores everything as nested records. Each
# record has an 8-byte header: a 2-byte field whose low nibble is recVer (0xF
# marks a *container* whose payload is more records) and whose high 12 bits are
# recInstance, a 2-byte recType, then a 4-byte recLen — all little-endian. Slide
# text lives in two atom types inside those containers.
_PPT_MAIN_STREAM = "PowerPoint Document"
_TEXT_BYTES_ATOM = 0x0FA8  # payload is Latin-1 (one byte per char)
_TEXT_CHARS_ATOM = 0x0FA0  # payload is UTF-16LE
_CONTAINER_VER = 0xF
_HEADER = struct.Struct("<HHI")


def extract_ppt_text(data: bytes) -> str:
    """Recover slide text from a raw 'PowerPoint Document' stream.

    Walks the record tree, recursing into containers and decoding every text
    atom it finds. Best-effort: the legacy binary format has no clean public
    parser, so this targets the text atoms rather than full fidelity.
    """
    parts: list[str] = []
    _walk(data, 0, len(data), parts, depth=0)
    text = "\n".join(p for p in parts if p.strip())
    # PPT uses \x0b (vertical tab) for soft line breaks and \r for paragraphs.
    return text.replace("\x0b", "\n").replace("\r", "\n")


def _walk(data: bytes, start: int, end: int, out: list[str], depth: int) -> None:
    # Guard against pathologically deep / malformed nesting.
    if depth > 32:
        return
    pos = start
    while pos + _HEADER.size <= end:
        ver_inst, rec_type, rec_len = _HEADER.unpack_from(data, pos)
        pos += _HEADER.size
        child_end = min(pos + rec_len, end)
        if (ver_inst & 0x0F) == _CONTAINER_VER:
            _walk(data, pos, child_end, out, depth + 1)
        elif rec_type == _TEXT_BYTES_ATOM:
            out.append(data[pos:child_end].decode("latin-1", errors="ignore"))
        elif rec_type == _TEXT_CHARS_ATOM:
            out.append(data[pos:child_end].decode("utf-16-le", errors="ignore"))
        pos = child_end


class PptExtractor:
    """Legacy binary PowerPoint (.ppt). Modern .pptx is handled by PptxExtractor."""

    async def extract(self, file_path: str) -> ExtractionResult:
        text = await asyncio.to_thread(self._read_sync, file_path)
        return ExtractionResult(text=text, source="native")

    @staticmethod
    def _read_sync(file_path: str) -> str:
        try:
            import olefile
        except ImportError as ex:
            logger.warning("olefile not available; cannot read %s: %s", file_path, ex)
            return ""

        try:
            if not olefile.isOleFile(file_path):
                # Not an OLE2 container — e.g. a .pptx misnamed .ppt, or corrupt.
                return ""
            ole = olefile.OleFileIO(file_path)
        except FileNotFoundError:
            return ""
        except Exception as ex:
            logger.warning("Failed to open PPT %s: %s", file_path, ex)
            return ""

        try:
            if not ole.exists(_PPT_MAIN_STREAM):
                return ""
            data = ole.openstream(_PPT_MAIN_STREAM).read()
        except Exception as ex:
            logger.warning("Failed to read PPT stream from %s: %s", file_path, ex)
            return ""
        finally:
            ole.close()

        return extract_ppt_text(data)
