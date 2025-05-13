import fitz  # PyMuPDF
import re
from collections import defaultdict, Counter
import Levenshtein  # pip install python-Levenshtein

# === Constants ===
HEADER_Y_THRESHOLD = 70
FOOTER_Y_THRESHOLD = 70
LEVENSHTEIN_SIMILARITY_THRESHOLD = 0.85
APPEARANCE_RATIO_THRESHOLD = 0.5  # Appears in ≥ 50% of pages

# === Utility Functions ===

def normalize_text(text):
    """Simplify text for comparison: lowercase, remove digits & punctuation."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def is_similar(a, b, threshold=LEVENSHTEIN_SIMILARITY_THRESHOLD):
    """Compare two strings for similarity using Levenshtein ratio."""
    if not a or not b:
        return False
    return Levenshtein.ratio(a, b) >= threshold

def group_similar_texts(text_list, threshold=LEVENSHTEIN_SIMILARITY_THRESHOLD):
    """Group similar normalized text blocks together."""
    groups = []
    for text in text_list:
        norm = normalize_text(text)
        found = False
        for group in groups:
            if any(is_similar(norm, g) for g in group):
                group.append(norm)
                found = True
                break
        if not found:
            groups.append([norm])
    return groups

# === Main Function ===

def extract_clean_text(pdf_path):
    """Extract text from PDF excluding repetitive headers/footers."""
    doc = fitz.open(pdf_path)
    header_footer_candidates = defaultdict(list)

    # Pass 1: Detect candidate headers and footers
    for page in doc:
        blocks = page.get_text("blocks")
        height = page.rect.height
        for block in blocks:
            x0, y0, x1, y1, text = *block[:4], block[4].strip()
            if not text:
                continue
            norm_text = normalize_text(text)
            if y0 < HEADER_Y_THRESHOLD:
                header_footer_candidates['header'].append(norm_text)
            elif y1 > height - FOOTER_Y_THRESHOLD:
                header_footer_candidates['footer'].append(norm_text)

    # Group and count repeated header/footer text
    frequent_headers = []
    frequent_footers = []
    num_pages = len(doc)

    for section in ['header', 'footer']:
        grouped = group_similar_texts(header_footer_candidates[section])
        group_counts = [
            sum(Counter(header_footer_candidates[section])[t] for t in group)
            for group in grouped
        ]
        frequent = [
            group[0] for group, count in zip(grouped, group_counts)
            if count / num_pages >= APPEARANCE_RATIO_THRESHOLD
        ]
        if section == 'header':
            frequent_headers = frequent
        else:
            frequent_footers = frequent

    # Pass 2: Clean pages
    cleaned_pages = []
    for page in doc:
        blocks = page.get_text("blocks")
        height = page.rect.height
        page_text = []
        for block in blocks:
            x0, y0, x1, y1, text = *block[:4], block[4].strip()
            if not text:
                continue
            norm_text = normalize_text(text)
            if (y0 < HEADER_Y_THRESHOLD and any(is_similar(norm_text, h) for h in frequent_headers)) or \
               (y1 > height - FOOTER_Y_THRESHOLD and any(is_similar(norm_text, f) for f in frequent_footers)):
                continue  # Skip header/footer
            page_text.append(text)
        cleaned_pages.append(" ".join(page_text))

    return "\n\n".join(cleaned_pages)
