from __future__ import annotations

import logging
import re
import struct

from ai_file_brain.core.models import ExtractionResult

logger = logging.getLogger(__name__)

# Legacy binary Word (.doc / Word 97-2003) keeps its text in the "WordDocument"
# stream, but the bytes are split into "pieces" (some compressed cp1252, some
# UTF-16LE) described by a piece table (the CLX) living in a separate table
# stream. We parse the File Information Block (FIB) to find the table, walk the
# piece table, and stitch the pieces back together — the same approach antiword
# and Apache POI use. Best-effort: malformed files degrade to "".
_WORD_STREAM = "WordDocument"
_FIB_IDENT = 0xA5EC  # wIdent at offset 0 for Word 97+

# Fixed offsets within the Word 97 FIB (the rgFcLcb97 array sits at a constant
# position for nFib >= 0x00C1, so fcClx/lcbClx are well-known).
_OFF_IDENT = 0x0000
_OFF_FLAGS = 0x000A  # bit 0x0200 (fWhichTblStm) picks 1Table vs 0Table
_OFF_FCCLX = 0x01A2
_OFF_LCBCLX = 0x01A6

_PCDT_CLXT = 0x02  # marks the piece-table block inside the CLX
_PRC_CLXT = 0x01  # marks a property-modifier block (skipped)
_FC_COMPRESSED = 0x40000000  # bit 30 of a piece FC: cp1252, 1 byte/char
_FC_MASK = 0x3FFFFFFF

# Field-instruction codes (\x13 begin … \x14 sep): keep the result, drop the
# instruction. Remaining low control chars become whitespace.
_FIELD_INSTRUCTION_RE = re.compile("\x13[^\x14\x15]*[\x14\x15]")
_PARAGRAPH_CHARS = str.maketrans({"\r": "\n", "\x0b": "\n", "\x0c": "\n", "\x07": "\n"})
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1f]")


def table_stream_name(word_stream: bytes) -> str:
    """Which OLE stream holds the piece table, per the FIB flags."""
    flags = struct.unpack_from("<H", word_stream, _OFF_FLAGS)[0]
    return "1Table" if (flags & 0x0200) else "0Table"


def extract_doc_text(word_stream: bytes, table_stream: bytes) -> str:
    """Reconstruct document text from the WordDocument + table streams.

    Returns "" for anything that isn't a parseable Word 97-2003 binary doc.
    """
    if len(word_stream) < _OFF_LCBCLX + 4:
        return ""
    if struct.unpack_from("<H", word_stream, _OFF_IDENT)[0] != _FIB_IDENT:
        return ""

    fc_clx = struct.unpack_from("<I", word_stream, _OFF_FCCLX)[0]
    lcb_clx = struct.unpack_from("<I", word_stream, _OFF_LCBCLX)[0]
    clx = table_stream[fc_clx : fc_clx + lcb_clx]
    plcfpcd = _find_piece_table(clx)
    if plcfpcd is None:
        return ""

    pieces = _read_pieces(word_stream, plcfpcd)
    return _clean("".join(pieces))


def _find_piece_table(clx: bytes) -> bytes | None:
    """Scan the CLX for its Pcdt block and return the raw plcfpcd bytes."""
    pos = 0
    n = len(clx)
    while pos < n:
        clxt = clx[pos]
        pos += 1
        if clxt == _PCDT_CLXT:
            if pos + 4 > n:
                return None
            lcb = struct.unpack_from("<I", clx, pos)[0]
            pos += 4
            return clx[pos : pos + lcb]
        if clxt == _PRC_CLXT:
            if pos + 2 > n:
                return None
            cb = struct.unpack_from("<H", clx, pos)[0]
            pos += 2 + cb
            continue
        return None  # unknown marker — give up rather than misread
    return None


def _read_pieces(word_stream: bytes, plcfpcd: bytes) -> list[str]:
    # plcfpcd = (n+1) CPs (4 bytes each) followed by n PCDs (8 bytes each).
    if len(plcfpcd) < 4:
        return []
    n = (len(plcfpcd) - 4) // 12
    if n <= 0:
        return []
    cps = [struct.unpack_from("<I", plcfpcd, i * 4)[0] for i in range(n + 1)]
    pcd_base = (n + 1) * 4

    out: list[str] = []
    for i in range(n):
        cch = cps[i + 1] - cps[i]
        if cch <= 0:
            continue
        fc_raw = struct.unpack_from("<I", plcfpcd, pcd_base + i * 8 + 2)[0]
        fc = fc_raw & _FC_MASK
        if fc_raw & _FC_COMPRESSED:
            start = fc // 2
            raw = word_stream[start : start + cch]
            out.append(raw.decode("cp1252", errors="ignore"))
        else:
            raw = word_stream[fc : fc + cch * 2]
            out.append(raw.decode("utf-16-le", errors="ignore"))
    return out


def _clean(text: str) -> str:
    text = _FIELD_INSTRUCTION_RE.sub("", text)
    text = text.translate(_PARAGRAPH_CHARS)
    text = _CONTROL_RE.sub(" ", text)
    # Collapse the runs of blank lines piece-stitching tends to produce.
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln.strip())


class DocExtractor:
    """Legacy binary Word (.doc). Modern .docx is handled by DocxExtractor."""

    async def extract(self, file_path: str) -> ExtractionResult:
        import asyncio

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
                return ""
            ole = olefile.OleFileIO(file_path)
        except FileNotFoundError:
            return ""
        except Exception as ex:
            logger.warning("Failed to open DOC %s: %s", file_path, ex)
            return ""

        try:
            if not ole.exists(_WORD_STREAM):
                return ""
            word_stream = ole.openstream(_WORD_STREAM).read()
            table_name = table_stream_name(word_stream)
            if not ole.exists(table_name):
                return ""
            table_stream = ole.openstream(table_name).read()
        except Exception as ex:
            logger.warning("Failed to read DOC streams from %s: %s", file_path, ex)
            return ""
        finally:
            ole.close()

        return extract_doc_text(word_stream, table_stream)
