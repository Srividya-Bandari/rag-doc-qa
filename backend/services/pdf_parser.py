import fitz  # PyMuPDF


def extract_text_by_page(pdf_bytes: bytes) -> list[dict]:
    """Extract text from a PDF, returning one entry per page.

    Uses sort=True so PyMuPDF reorders glyphs by position before returning,
    giving correct reading order for normal selectable-text PDFs.
    Pages with no extractable text are omitted.

    Each entry: {"page_number": int (1-based), "text": str}
    Note: switch to get_text("blocks") if multi-column extraction proves broken.
    """
    pages: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            text = doc[page_index].get_text("text", sort=True).strip()
            if text:
                pages.append({"page_number": page_index + 1, "text": text})
    return pages

