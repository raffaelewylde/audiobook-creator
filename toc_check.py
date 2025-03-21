import fitz
import re
from pathlib import Path

pdf = Path("path_to_your_pdf.pdf")
doc = fitz.open(pdf)
toc = doc.get_toc()


def find_first_match(toc, keywords):
    results = {}
    for keyword in keywords:
        pattern = re.compile(keyword, re.IGNORECASE)
        match = next((item for item in toc if pattern.search(item[1])), None)
        if match:
            results[keyword] = match[2]
        else:
            results[keyword] = None
    return results
