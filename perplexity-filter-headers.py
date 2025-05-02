import fitz  # PyMuPDF


def extract_text_without_headers_footers(pdf_path, header_height=50, footer_height=100):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        # Define the clip area (exclude header and footer regions)
        clip_rect = page.rect  # Full page rectangle
        clip_rect.y0 += header_height  # Skip header
        clip_rect.y1 -= footer_height  # Skip footer
        # Extract text from the clipped region
        text += page.get_text(clip=clip_rect, sort=True)
    return text


# Adjust header_height and footer_height based on your PDF’s layout
# Use page.rect to get the page dimensions and modify y0/y1 to exclude unwanted areas


# Version 2 - use font attributes
def filter_blocks(pdf_path, max_header_font=9, min_footer_font=8):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        blocks = page.get_text("dict", sort=True)["blocks"]
        for block in blocks:
            # Skip blocks in header/footer regions or with suspicious fonts
            if (
                block["bbox"][1] < 50  # Header region (y0 < 50)
                or block["bbox"][3] > page.rect.height - 100  # Footer region
                or block["size"] < max_header_font  # Smaller font (header)
                or block["size"] < min_footer_font
            ):
                continue
            text += block["text"] + "\n"
    return text
