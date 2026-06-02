from __future__ import annotations

import asyncio
import logging

from ai_file_brain.core.models import ExtractionResult

logger = logging.getLogger(__name__)


class XlsExtractor:
    """Legacy binary Excel (.xls / BIFF). Modern .xlsx is handled by XlsxExtractor."""

    async def extract(self, file_path: str) -> ExtractionResult:
        text = await asyncio.to_thread(self._read_sync, file_path)
        return ExtractionResult(text=text, source="native")

    @staticmethod
    def _read_sync(file_path: str) -> str:
        try:
            import xlrd
        except ImportError as ex:
            logger.warning("xlrd not available; cannot read %s: %s", file_path, ex)
            return ""

        try:
            book = xlrd.open_workbook(file_path)
        except FileNotFoundError:
            return ""
        except Exception as ex:
            # Password-protected, malformed, or an .xlsx misnamed as .xls
            # (xlrd 2.x refuses .xlsx outright).
            logger.warning("Failed to open XLS %s: %s", file_path, ex)
            return ""

        parts: list[str] = []
        for sheet in book.sheets():
            rows: list[str] = []
            for r in range(sheet.nrows):
                cells = [str(v) for v in sheet.row_values(r) if str(v).strip()]
                if cells:
                    rows.append("\t".join(cells))
            if rows:
                parts.append(f"# {sheet.name}")
                parts.extend(rows)

        return "\n".join(parts)
