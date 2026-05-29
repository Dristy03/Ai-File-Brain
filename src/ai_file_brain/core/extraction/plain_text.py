from __future__ import annotations

import aiofiles

from ai_file_brain.core.models import ExtractionResult


class PlainTextExtractor:
    async def extract(self, file_path: str) -> ExtractionResult:
        async with aiofiles.open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            text = await f.read()
        return ExtractionResult(text=text, source="native")
