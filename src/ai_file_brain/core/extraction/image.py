from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import numpy as np

from ai_file_brain.core.extraction.ocr import ocr_image
from ai_file_brain.core.models import ExtractionResult

logger = logging.getLogger(__name__)


class ImageExtractor:
    async def extract(self, file_path: str) -> ExtractionResult:
        frames = await asyncio.to_thread(self._load_frames_sync, file_path)
        if not frames:
            return ExtractionResult(text="", source="ocr")
        page_texts: list[str] = []
        for frame in frames:
            text = await ocr_image(frame)
            if text.strip():
                page_texts.append(text)
        return ExtractionResult(text="\n".join(page_texts), source="ocr")

    @staticmethod
    def _load_frames_sync(file_path: str) -> list[np.ndarray]:
        try:
            from PIL import Image, ImageSequence
        except ImportError as ex:
            logger.warning("Pillow not available; cannot OCR %s: %s", file_path, ex)
            return []

        try:
            img = Image.open(file_path)
            img.load()
        except FileNotFoundError:
            return []
        except Exception as ex:
            logger.warning("Failed to open image %s: %s", file_path, ex)
            return []

        try:
            if getattr(img, "is_animated", False):
                # Animated formats (e.g. GIF): OCR only the first frame.
                return [_to_rgb_ndarray(img)]
            n_frames = getattr(img, "n_frames", 1) or 1
            if n_frames <= 1:
                return [_to_rgb_ndarray(img)]
            # Multi-page (e.g. multi-page TIFF): OCR every page.
            frames: list[np.ndarray] = []
            for page in ImageSequence.Iterator(img):
                frames.append(_to_rgb_ndarray(page))
            return frames
        finally:
            with contextlib.suppress(Exception):
                img.close()


def _to_rgb_ndarray(img: Any) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)
