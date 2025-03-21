import re
import fitz  # PyMuPDF
import pandas as pd
from unidecode import unidecode
import numpy as np

doc = fitz.open("example.pdf")  # Open a sample PDF

block_dict = {}
page_num = 1
for page in doc:  # Iterate all pages in the document
    file_dict = page.get_text("dict")  # Get the page dictionary
    block = file_dict["blocks"]  # Get the block information
    block_dict[page_num] = block  # Store in block dictionary
    page_num += 1  # Increase the page value by 1

# Fix: Ensure column names match the number of appended values
spans = pd.DataFrame(
    columns=[
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "text",
        "is_upper",
        "is_bold",
        "span_font",
        "font_size",
    ]
)
rows = []

for page_num, blocks in block_dict.items():
    for block in blocks:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    xmin, ymin, xmax, ymax = list(span["bbox"])
                    font_size = span["size"]
                    text = unidecode(span["text"])
                    span_font = span["font"]
                    is_upper = False
                    is_bold = False

                    if "bold" in span_font.lower():
                        is_bold = True

                    if re.sub(r"[\(\[].*?[\)\]]", "", text).isupper():
                        is_upper = True

                    if text.replace(" ", "") != "":
                        rows.append(
                            (
                                xmin,
                                ymin,
                                xmax,
                                ymax,
                                text,
                                is_upper,
                                is_bold,
                                span_font,
                                font_size,
                            )
                        )

# Fix: Ensure dataframe columns match the appended row structure
span_df = pd.DataFrame(
    rows,
    columns=[
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "text",
        "is_upper",
        "is_bold",
        "span_font",
        "font_size",
    ],
)


span_scores = []
special = r"[(_:/,#%\=@)]"

for index, span_row in span_df.iterrows():
    score = round(span_row.font_size)
    text = span_row.text

    if not re.search(special, text):
        if span_row.is_bold:
            score += 1
        if span_row.is_upper:
            score += 1

    span_scores.append(score)

if span_scores:  # Check if list is not empty
    values, counts = np.unique(span_scores, return_counts=True)
    style_dict = {value: count for value, count in zip(values, counts)}
    sorted_style_dict = sorted(style_dict.items(), key=lambda x: x[1])
else:
    style_dict = {}
    sorted_style_dict = []

print("style_dict:", style_dict)
print("sorted_style_dict:", sorted_style_dict)
p_size = max(style_dict, key=style_dict.get)
idx = 0
tag = {}
for size in sorted(values, reverse=True):
    idx += 1
    if size == p_size:
        idx = 0
        tag[size] = "p"
    if size > p_size:
        tag[size] = "h{0}".format(idx)
    if size < p_size:
        tag[size] = "s{0}".format(idx)

print(tag)
span_tags = [tag[score] for score in span_scores]
span_df["tag"] = span_tags
headings_list = []
text_list = []
tmp = []
heading = ""
for index, span_row in span_df.iterrows():
    text = span_row.text
    tag = span_row.tag
    if "h" in tag:
        headings_list.append(text)
        text_list.append("\n".join(tmp))
        tmp = []
        heading = text
    else:
        tmp.append(text)
text_list.append("\n".join(tmp))
text_list = text_list[1:]
text_df = pd.DataFrame(zip(headings_list, text_list), columns=["heading", "content"])
