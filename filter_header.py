import fitz  # PyMuPDF
import re
import spacy
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def clean_header_footer(candidates):
    if not candidates:
        return ""

    # Tokenize text, remove numbers and stopwords
    processed = []
    for text in candidates:
        text = re.sub(r"\d+", "", text).strip()  # Remove numbers
        doc = nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.is_alpha
        ]
        processed.append(" ".join(tokens))

    # Find most common phrase
    common_part = Counter(processed).most_common(1)
    return common_part[0][0] if common_part else ""


def extract_text_without_headers_footers(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = []

    header_candidates = []
    footer_candidates = []

    for page in doc:
        blocks = page.get_text("blocks")  # Extract text blocks with positions
        blocks = sorted(blocks, key=lambda b: b[1])  # Sort by vertical position

        page_height = page.rect.height

        # Define threshold for headers/footers (top and bottom 5-10%)
        header_threshold = page_height * 0.1
        footer_threshold = page_height * 0.9

        page_body = []

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block

            if y1 < header_threshold:
                header_candidates.append(text.strip())
            elif y0 > footer_threshold:
                footer_candidates.append(text.strip())
            else:
                page_body.append(text.strip())

        page_texts.append(" ".join(page_body))

    # Detect common headers and footers using spaCy processing
    header_text = clean_header_footer(header_candidates)
    footer_text = clean_header_footer(footer_candidates)

    # Remove detected headers and footers from the pages
    final_text = []
    for text in page_texts:
        if header_text in text:
            text = text.replace(header_text, "").strip()
        if footer_text in text:
            text = text.replace(footer_text, "").strip()
        text = re.sub(r"^\d+\s*", "", text)  # Remove leading page numbers
        final_text.append(text)

    return "\n".join(final_text), header_text, footer_text


# Example usage
pdf_path = "invent_your_own_computer_games_with_python.pdf"
clean_text, detected_header, detected_footer = extract_text_without_headers_footers(
    pdf_path
)
print("Detected Header:", detected_header)
print("Detected Footer:", detected_footer)
print(clean_text[:1000])  # Print first 1000 characters to verify
