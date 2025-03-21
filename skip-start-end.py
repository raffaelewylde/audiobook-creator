import fitz  # PyMuPDF
import re


def extract_main_content(pdf_path, output_txt_path):
    doc = fitz.open(pdf_path)

    start_reading = False
    end_reached = False
    in_toc = False  # Flag to track if we're in the Table of Contents
    text_output = []

    # Define patterns for start and end
    toc_keywords = [r"\bContents\b", r"\bTable of Contents\b"]
    start_keywords = [
        r"\b(?:Chapter|Part)+\s+(?:[IVXLCDM]+|[0-9]+|(One)+)",
        r"\bIntroduction\b",
        r"\bPrologue\b",
        "One",
        "1",
        r"\b\d+(\.\d+)+\s+\w+\b",
    ]
    end_keywords = [r"\bGlossary\b", r"\bAppendix\b", r"\bIndex\b", r"\bReferences\b"]

    for page in doc:
        text = page.get_text("text")
        if not text.strip():
            continue  # Skip empty pages

        lines = text.split("\n")  # Split page text into individual lines

        # Detect and handle Table of Contents section
        if not start_reading:
            for pattern in toc_keywords:
                if re.search(pattern, text, re.IGNORECASE):
                    in_toc = True  # Mark ToC start
                    break

        if in_toc:
            # Heuristic: If most lines are short, assume it's still ToC
            short_lines = sum(1 for line in lines if len(line.strip()) < 50)
            if short_lines > len(lines) * 0.6:  # If >60% of lines are short
                continue  # Skip ToC pages
            else:
                in_toc = False  # Exit ToC once we see full sentences

        # Detect actual start of content (only after skipping ToC)
        if not start_reading and not in_toc:
            first_three_lines = lines[:3]  # Extract first three lines
            for pattern in start_keywords:
                if any(
                    re.search(pattern, line, re.IGNORECASE)
                    for line in first_three_lines
                ):
                    # Ensure the text has full paragraphs before starting
                    if len(lines) > 5:  # Require at least 5 full lines
                        start_reading = True
                        break

        # Save text if we're in the main content
        if start_reading and not end_reached:
            text_output.append(text)

            # Check for end markers
            for pattern in end_keywords:
                if re.search(pattern, text, re.IGNORECASE):
                    end_reached = True
                    break

    # Save extracted text to a file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_output))


# Example usage
extract_main_content("ebook.pdf", "output.txt")
