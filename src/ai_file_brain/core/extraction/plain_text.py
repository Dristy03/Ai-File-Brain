from __future__ import annotations

import logging

import aiofiles

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.models import ExtractionResult

logger = logging.getLogger(__name__)


class PlainTextExtractor:
    async def extract(self, file_path: str) -> ExtractionResult:
        cap = AiFileBrainSettings().max_extracted_chars
        async with aiofiles.open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            if cap and cap > 0:
                # Read one past the cap so we can tell whether truncation happened
                # without pulling a multi-hundred-MB file fully into memory.
                text = await f.read(cap + 1)
                if len(text) > cap:
                    logger.info(
                        "Truncating %s to first %d chars (max_extracted_chars)",
                        file_path,
                        cap,
                    )
                    text = text[:cap]
            else:
                text = await f.read()
        return ExtractionResult(text=text, source="native")
