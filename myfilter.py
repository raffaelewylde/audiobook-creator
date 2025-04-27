import fitz
import re


def toc_chapter_1_check(pdf_file):
    doc = fitz.open(pdf_file)
    pattern1 = r"\bchapter 1(?!\d)"
    pattern2 = r"\b(1|one)[-:\.,]"
    toc = doc.get_toc()
    if toc:
        for item in toc:
            match1 = re.search(pattern1, item[1], re.IGNORECASE)
            match2 = re.search(pattern2, item[1], re.IGNORECASE)
            if match1 or match2:
                return item[2]
