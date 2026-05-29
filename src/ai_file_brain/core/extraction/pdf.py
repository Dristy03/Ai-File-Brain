from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import numpy as np
from pypdf import PdfReader

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.extraction.ocr import ocr_image
from ai_file_brain.core.models import ExtractionResult

logger = logging.getLogger(__name__)


class PdfExtractor:
    async def extract(self, file_path: str) -> ExtractionResult:
        native_pages = await asyncio.to_thread(self._read_native_pages, file_path)
        if native_pages is None:
            return ExtractionResult(text="", source="native")

        total_native_chars = sum(len(p.strip()) for p in native_pages)
        settings = AiFileBrainSettings()

        if (
            not settings.ocr_enabled
            or total_native_chars >= settings.pdf_ocr_min_native_chars
        ):
            return ExtractionResult(
                text="\n".join(p for p in native_pages if p),
                source="native",
            )

        # Fallback path: per-page OCR for sparse pages.
        rendered = await asyncio.to_thread(
            self._render_sparse_pages,
            file_path,
            native_pages,
            settings.pdf_ocr_per_page_min_chars,
            settings.pdf_ocr_render_dpi,
        )
        if rendered is None:
            # PyMuPDF could not open the doc; fall back to whatever native text we had.
            return ExtractionResult(
                text="\n".join(p for p in native_pages if p),
                source="native",
            )

        page_images, ocr_indices = rendered
        ocr_texts: dict[int, str] = {}
        for idx in ocr_indices:
            ocr_texts[idx] = await ocr_image(page_images[idx])

        final_pages: list[str] = []
        used_ocr = False
        used_native = False
        for i, native in enumerate(native_pages):
            if i in ocr_texts:
                final_pages.append(ocr_texts[i])
                used_ocr = True
            else:
                final_pages.append(native)
                if native.strip():
                    used_native = True

        if used_ocr and used_native:
            source = "mixed"
        elif used_ocr:
            source = "ocr"
        else:
            source = "native"

        return ExtractionResult(
            text="\n".join(p for p in final_pages if p),
            source=source,
        )

    @staticmethod
    def _read_native_pages(file_path: str) -> list[str] | None:
        try:
            reader = PdfReader(file_path)
        except FileNotFoundError:
            return None
        except Exception as ex:
            logger.warning("Failed to open PDF %s: %s", file_path, ex)
            return None
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as ex:
                logger.warning("Failed to extract page %d of %s: %s", i, file_path, ex)
                pages.append("")
        return pages

    @staticmethod
    def _render_sparse_pages(
        file_path: str,
        native_pages: list[str],
        per_page_min_chars: int,
        dpi: int,
    ) -> tuple[list[Any], list[int]] | None:
        try:
            import pymupdf
        except ImportError as ex:
            logger.warning("PyMuPDF not available; cannot OCR PDF %s: %s", file_path, ex)
            return None

        try:
            doc = pymupdf.open(file_path)
        except Exception as ex:
            logger.warning("PyMuPDF could not open %s: %s", file_path, ex)
            return None

        try:
            page_count = doc.page_count
            # Use the larger of pypdf's count and PyMuPDF's count to size the list.
            n = max(page_count, len(native_pages))
            page_images: list[Any] = [None] * n
            ocr_indices: list[int] = []
            zoom = dpi / 72.0
            mat = pymupdf.Matrix(zoom, zoom)
            for i in range(page_count):
                native_text = native_pages[i] if i < len(native_pages) else ""
                if len(native_text.strip()) >= per_page_min_chars:
                    continue
                try:
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    page_images[i] = img.copy()  # detach from pix.samples buffer
                    ocr_indices.append(i)
                except Exception as ex:
                    logger.warning(
                        "Failed to render page %d of %s for OCR: %s", i, file_path, ex
                    )
            return page_images, ocr_indices
        finally:
            with contextlib.suppress(Exception):
                doc.close()
