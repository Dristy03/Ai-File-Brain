from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from ai_file_brain.config import AiFileBrainSettings

logger = logging.getLogger(__name__)

_engine: Any = None
_engine_failed: bool = False
_engine_lock: asyncio.Lock | None = None


async def ocr_image(
    img: np.ndarray, settings: AiFileBrainSettings | None = None
) -> str:
    """Run OCR on a numpy image array. Returns recognised text joined by newlines.

    Returns empty string when OCR is disabled, the engine fails to load, or no
    text is recognised. Never raises.

    ``settings`` may be passed in by callers that already built one (e.g. the PDF
    extractor, which OCRs many pages per file) to avoid re-parsing settings.toml
    on every page; it's constructed on demand when omitted.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return ""
    if settings is None:
        settings = AiFileBrainSettings()
    if not settings.ocr_enabled:
        return ""
    engine = await _get_engine(settings)
    if engine is None:
        return ""
    return await asyncio.to_thread(_run_ocr_sync, engine, img)


async def _get_engine(settings: AiFileBrainSettings):
    global _engine, _engine_failed, _engine_lock
    if _engine is not None:
        return _engine
    if _engine_failed:
        return None
    if _engine_lock is None:
        _engine_lock = asyncio.Lock()
    async with _engine_lock:
        if _engine is not None:
            return _engine
        if _engine_failed:
            return None
        try:
            _engine = await asyncio.to_thread(_init_engine_sync, settings)
        except Exception as ex:
            logger.warning("Failed to initialize RapidOCR: %s", ex)
            _engine_failed = True
            return None
    return _engine


def _init_engine_sync(settings: AiFileBrainSettings):
    from rapidocr_onnxruntime import RapidOCR

    if settings.ocr_use_gpu:
        import onnxruntime as ort

        if "DmlExecutionProvider" in ort.get_available_providers():
            try:
                # DirectML for all three OCR models (detect / classify / recognize).
                engine = RapidOCR(
                    **{"Det.use_dml": True, "Cls.use_dml": True, "Rec.use_dml": True}
                )
                logger.info(
                    "Loaded RapidOCR with DirectML GPU acceleration (languages=%s)",
                    settings.ocr_languages,
                )
                return engine
            except Exception as ex:
                # Runtime present but the GPU couldn't init (driver/busy/OOM). Don't
                # disable OCR — fall through and load on CPU instead.
                logger.warning(
                    "DirectML GPU OCR failed to initialize (%s); falling back to CPU.",
                    ex,
                )
        else:
            logger.warning(
                "ocr_use_gpu=True but DmlExecutionProvider is unavailable; "
                "install onnxruntime-directml to use the GPU. Falling back to CPU."
            )

    logger.info("Loading RapidOCR on CPU (languages=%s)", settings.ocr_languages)
    return RapidOCR()


def _run_ocr_sync(engine, img: np.ndarray) -> str:
    try:
        result, _elapse = engine(img)
    except Exception as ex:
        logger.warning("RapidOCR failed: %s", ex)
        return ""
    if not result:
        return ""
    lines: list[str] = []
    for item in result:
        if len(item) >= 2 and item[1]:
            lines.append(str(item[1]))
    return "\n".join(lines)


def _reset_for_tests() -> None:
    """Reset module state. Call from tests that toggle ocr_enabled."""
    global _engine, _engine_failed
    _engine = None
    _engine_failed = False
