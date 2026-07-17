"""OCR engine behind an interface: real local OCR by default when
`OCR_BACKEND=tesseract` (pytesseract + the system `tesseract` binary, no
cloud key), a deterministic fake for offline tests and the default —
same real/fake split as `app.knowledge.embeddings`/`app.llm.interface`.
"""

from __future__ import annotations

from typing import Protocol

from app.config import settings

# Deterministic canned OCR text for offline tests/demos — shaped like a real
# lab requisition form so the entity-extraction prompt has something
# plausible to parse without a system OCR binary or an image at all.
FAKE_OCR_TEXT = (
    "LAB REQUISITION FORM\n"
    "Patient: Jane Doe\n"
    "Date: 2026-06-01\n"
    "Ordering Physician: Dr. Smith\n"
    "Tests Requested: GLU, HBA1C\n"
)


class OcrEngine(Protocol):
    def extract_text(self, image_bytes: bytes) -> str: ...


class FakeOcrEngine:
    """Returns the same canned text regardless of input — exercises the
    extraction/validation/persistence plumbing without a system OCR binary."""

    def extract_text(self, image_bytes: bytes) -> str:
        return FAKE_OCR_TEXT


class TesseractOcrEngine:
    """Real OCR via `pytesseract`, which shells out to the system `tesseract`
    binary (`brew install tesseract` locally). Not exercised by `make test`."""

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        import pytesseract

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._pytesseract = pytesseract

    def extract_text(self, image_bytes: bytes) -> str:
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        return self._pytesseract.image_to_string(image)


def get_ocr_engine() -> OcrEngine:
    if settings.ocr_backend == "tesseract":
        return TesseractOcrEngine(tesseract_cmd=settings.tesseract_cmd)
    return FakeOcrEngine()
