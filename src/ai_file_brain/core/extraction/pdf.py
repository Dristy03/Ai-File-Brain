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

        # Fallback path: per-page OCR for sparse pages. We render and OCR ONE page
        # at a time (rather than rasterizing every sparse page up front), so memory
        # stays at ~one page-image regardless of document length. At 220 DPI a
        # single A4 page is ~25 MB in RAM; holding all of them for a 500-page scan
        # would be >10 GB and OOM the process — which is the whole reason a tight
        # file-size cap used to be necessary.
        ocr = await self._ocr_sparse_pages(
            file_path,
            native_pages,
            settings.pdf_ocr_per_page_min_chars,
            settings.pdf_ocr_render_dpi,
            settings.pdf_ocr_max_pages,
        )
        if ocr is None:
            # PyMuPDF could not open the doc; fall back to whatever native text we had.
            return ExtractionResult(
                text="\n".join(p for p in native_pages if p),
                source="native",
            )

        ocr_texts, page_count = ocr
        final_pages: list[str] = []
        used_ocr = False
        used_native = False
        for i in range(max(len(native_pages), page_count)):
            native = native_pages[i] if i < len(native_pages) else ""
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

    async def _ocr_sparse_pages(
        self,
        file_path: str,
        native_pages: list[str],
        per_page_min_chars: int,
        dpi: int,
        max_pages: int,
    ) -> tuple[dict[int, str], int] | None:
        """Render + OCR each text-sparse page one at a time, freeing the page
        image before moving to the next. Returns ({page_index: ocr_text}, page_count),
        or None if PyMuPDF can't open the document.

        Only ~one page raster is alive at a time, so peak memory is independent of
        page count. ``max_pages`` (0 = unlimited) caps how many pages get OCR'd so a
        pathological multi-thousand-page scan can't pin the CPU indefinitely.
        """
        opened = await asyncio.to_thread(self._open_doc, file_path)
        if opened is None:
            return None
        doc, page_count = opened

        try:
            indices = [
                i
                for i in range(page_count)
                if len((native_pages[i] if i < len(native_pages) else "").strip())
                < per_page_min_chars
            ]
            if max_pages > 0 and len(indices) > max_pages:
                logger.info(
                    "Capping OCR at %d of %d sparse pages for %s (pdf_ocr_max_pages)",
                    max_pages,
                    len(indices),
                    file_path,
                )
                indices = indices[:max_pages]

            ocr_texts: dict[int, str] = {}
            for i in indices:
                img = await asyncio.to_thread(self._render_page, doc, i, dpi, file_path)
                if img is None:
                    continue
                ocr_texts[i] = await ocr_image(img)
                del img  # release the ~25 MB raster before rendering the next page
            return ocr_texts, page_count
        finally:
            await asyncio.to_thread(self._close_doc, doc)

    @staticmethod
    def _open_doc(file_path: str) -> tuple[Any, int] | None:
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
        return doc, doc.page_count

    @staticmethod
    def _render_page(doc: Any, i: int, dpi: int, file_path: str) -> Any:
        import pymupdf

        try:
            zoom = dpi / 72.0
            mat = pymupdf.Matrix(zoom, zoom)
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            return img.copy()  # detach from pix.samples buffer
        except Exception as ex:
            logger.warning(
                "Failed to render page %d of %s for OCR: %s", i, file_path, ex
            )
            return None

    @staticmethod
    def _close_doc(doc: Any) -> None:
        with contextlib.suppress(Exception):
            doc.close()
