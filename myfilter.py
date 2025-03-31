import fitz
import re


def toc_existence_check(pdf_file):
    doc = fitz.open(pdf_file)
    pattern = r"\bchapter 1(?!\d)"
    toc = doc.get_toc()
    if toc:
        for item in toc:
            match = re.search(pattern, item[1], re.IGNORECASE)
            if match:
                return item[2]
