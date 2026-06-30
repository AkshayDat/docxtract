import io
from PIL import Image

try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def ocr_page(pdf_path: str, page_num: int, dpi: int = 300) -> str:
    """Run OCR on a single page of a scanned PDF. Returns extracted text."""
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR dependencies not installed. Install pdf2image and pytesseract, "
            "and ensure Tesseract and Poppler are on your system PATH."
        )
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=dpi)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def page_to_image(pdf_path: str, page_num: int, dpi: int = 200) -> Image.Image:
    """Convert a single PDF page to a PIL Image."""
    if not OCR_AVAILABLE:
        raise RuntimeError("pdf2image is required for page_to_image.")
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=dpi)
    return images[0] if images else None


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 string for OpenAI vision API."""
    import base64
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
