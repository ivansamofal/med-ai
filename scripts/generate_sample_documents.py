"""One-off generator for sample "scanned" lab requisition form images used to
demo the OCR pipeline (`POST /documents/ocr`). Drawn with Pillow's built-in
bitmap font rather than a real scan — good enough for Tesseract to read back,
and needs no external image assets. Synthetic, non-clinical, same spirit as
`generate_sample_guidelines.py`.
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demo" / "sample_documents"

FORMS = {
    "requisition_jane_doe": [
        "LAB REQUISITION FORM",
        "",
        "Patient: Jane Doe",
        "Date: 2026-06-01",
        "Ordering Physician: Dr. Smith",
        "Tests Requested: GLU, HBA1C",
    ],
    "requisition_john_roe": [
        "LAB REQUISITION FORM",
        "",
        "Patient: John Roe",
        "Date: 2026-06-15",
        "Ordering Physician: Dr. Patel",
        "Tests Requested: K, CR, EGFR",
    ],
    "requisition_missing_fields": [
        "LAB REQUISITION FORM",
        "",
        "Patient: ",
        "Date: 2026-06-20",
        "Ordering Physician: Dr. Lee",
        "Tests Requested: XYZ",
    ],
}


def build_image(lines: list[str]) -> Image.Image:
    image = Image.new("RGB", (600, 400), color="white")
    draw = ImageDraw.Draw(image)
    y = 30
    for line in lines:
        draw.text((30, y), line, fill="black")
        y += 30
    return image


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, lines in FORMS.items():
        image = build_image(lines)
        output_path = OUTPUT_DIR / f"{filename}.png"
        image.save(output_path)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
