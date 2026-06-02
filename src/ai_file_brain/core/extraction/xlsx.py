from __future__ import annotations

import asyncio
import logging

from ai_file_brain.core.models import ExtractionResult

logger = logging.getLogger(__name__)


class XlsxExtractor:
    async def extract(self, file_path: str) -> ExtractionResult:
        text = await asyncio.to_thread(self._read_sync, file_path)
        return ExtractionResult(text=text, source="native")

    @staticmethod
    def _read_sync(file_path: str) -> str:
        try:
            from openpyxl import load_workbook
        except ImportError as ex:
            logger.warning("openpyxl not available; cannot read %s: %s", file_path, ex)
            return ""

        # read_only keeps memory flat on big sheets; data_only returns the last
        # cached cell *values* instead of formula strings.
        try:
            wb = load_workbook(file_path, read_only=True, data_only=True)
        except FileNotFoundError:
            return ""
        except Exception as ex:
            # Password-protected, malformed, or a legacy .xls misnamed as .xlsx.
            logger.warning("Failed to open XLSX %s: %s", file_path, ex)
            return ""

        parts: list[str] = []
        try:
            for ws in wb.worksheets:
                rows: list[str] = []
                for row in ws.iter_rows(values_only=True):
                    cells = [str(v) for v in row if v is not None and str(v).strip()]
                    if cells:
                        rows.append("\t".join(cells))
                if rows:
                    # Title the sheet's block so retrieval keeps per-sheet context.
                    parts.append(f"# {ws.title}")
                    parts.extend(rows)
        finally:
            wb.close()

        return "\n".join(parts)
