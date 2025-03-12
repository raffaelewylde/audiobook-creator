import asyncio
import logging
import os
import random
import re
import sys
from pathlib import Path

import aiofiles
import pymupdf
import pytesseract
from aiohttp import ClientError
from deepgram import (
    ClientOptionsFromEnv,
    DeepgramClient,
    SpeakOptions,
)
from openai import AsyncOpenAI
from pdf2image import convert_from_path
from pydub import AudioSegment
from tqdm.asyncio import tqdm

# CLAUSE_BOUNDARIES = r"\.|\?|!|;|, (?:and|but|or|nor|for|yet|so)"
# CLAUSE_BOUNDARIES = r"(?<=[.?!;])\s+|(?<!\w)\n"
CLAUSE_BOUNDARIES = r"""(?x)          # Enable verbose mode for clarity
    (?<=        # Positive lookbehind
        [.!?]   # After period, exclamation, or question mark
    )
    \s+         # Followed by whitespace
    (?=[A-Z])   # Followed by capital letter (likely new sentence)
    |           # OR
    (?<=[:;])   # After colon or semicolon
    \s+         # Followed by whitespace
    |           # OR
    ,\s+        # Comma followed by whitespace
    (?=         # Followed by common conjunctions
        (?:and|but|or|nor|for|yet|so)\s
    )
    |           # OR
    \n{2,}      # Two or more newlines (paragraph breaks)
"""
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    AsyncOpenAI.api_key = api_key
else:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


def setup_logging():
    """
        Configures logging to output messages to both the console and a rotating file.
        Logs INFO messages to stdout and DEBUG+ messages to a file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set lowest level to capture all messages

    # Prevent duplicate log entries
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Logs basic messages to console

    file_handler = logging.FileHandler("app.log", "w", "utf-8")
    file_handler.setLevel(logging.DEBUG)  # Logs everything to file

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s - Line: %(lineno)d"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False  # Prevent double logging

    return logger


logger = setup_logging()


def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.point(lambda x: 0 if x < 128 else 255)  # Binarize
    return image


# Wew're copying a couple of functions from PyMuPDF to assist with the 
# text_extraction function we'll define in a moment
def page_layout(page, textout, GRID, fontsize, noformfeed, skip_empty, flags):
    left = page.rect.width  # left most used coordinate
    right = 0  # rightmost coordinate
    rowheight = page.rect.height  # smallest row height in use
    chars = []  # all chars here
    rows = set()  # bottom coordinates of lines
    if noformfeed:
        eop = b"\n"
    else:
        eop = bytes([12])


def clean_text(text, header_pattern, footer_pattern):
    """
    Removes header and footer text from the given text.
    """
    if header_pattern:
        text = text.replace(header_pattern, "")
    if footer_pattern:
        text = text.replace(footer_pattern, "")

    return text.strip()


def identify_headers_footers(doc):
    """
    Identify common header and footer patterns in the PDF document.
    """
    header_patterns = []
    footer_patterns = []
    num_pages = len(doc)

    for page_num in range(min(10, num_pages)):  # Analyze the first 10 pages or less
        page = doc[page_num]
        text = page.get_text(sort=True)
        lines = text.split("\n")

        if len(lines) > 2:
            header_patterns.append(lines[0])
            footer_patterns.append(lines[-1])

    # Identify common patterns
    header_pattern = (
        max(set(header_patterns), key=header_patterns.count)
        if header_patterns
        else None
    )
    footer_pattern = (
        max(set(footer_patterns), key=footer_patterns.count)
        if footer_patterns
        else None
    )

    return header_pattern, footer_pattern


def extract_text_from_pdf(pdf_path):
    """
    The function `extract_text_from_pdf` reads a PDF file and extracts its text content, skipping the
    beginning pages until it encounters the keywords "preface", "introduction", or "chapter 1".

    :param pdf_path: The `pdf_path` parameter in the `extract_text_from_pdf` function is a string that
    represents the file path to the PDF file from which you want to extract text
    :return: The function `extract_text_from_pdf` returns the extracted text from the PDF file located
    at the `pdf_path` provided as input.
    """
    """
    logger.info("Converting your pdf, %s, to plain text", pdf_path)
    filetype = type(pdf_path)
    logger.debug("Type for pdf is %s", filetype)

    keywords = ["preface", "introduction", "chapter 1"]
    extracting = False
    toc_passed = False
    in_toc = False
    toc_keyword = None
    extracted_text = []
    logger.info("Ok we've set up our keyword and flags, now lets test extraction")
    """
    try:
        doc = pymupdf.open(pdf_path)
        header_pattern, footer_pattern = identify_headers_footers(doc)
        logger.debug("Identified header pattern: %s", header_pattern)
        logger.debug("Identified footer pattern: %s", footer_pattern)

        for page in doc:
            text = page.get_text(sort=True)  # type: ignore
            logger.debug("Text on page %s: %s", page.number, text)
            logger.info("Looking for keywords in text on page: %s", page.number)
            if not text:
                logger.warning("No text found on page %s", page.number)
                continue

            # Let's clean the text by removing header and footer text
            text = clean_text(text, header_pattern, footer_pattern)
            logger.debug("Removing any header and footer text from text")

            # Check if we are in the table of contents
            if "table of contents" or "contents" in text.lower():
                in_toc = True
                logger.debug("Found table of contents, setting in_toc to True")

            # Check if any of the keywords are in the text
            for keyword in keywords:
                if keyword in text.lower():
                    logger.debug(
                        "Found keyword '%s' in text. Let's check if we're in the TOC or not.",
                        keyword,
                    )
                    if in_toc:
                        toc_keyword = keyword
                        in_toc = False
                        logger.debug(
                            "Keyword '%s' found in TOC, setting toc_keyword to '%s'",
                            keyword,
                            toc_keyword,
                        )
                    elif toc_keyword == keyword:
                        extracting = True
                        logger.debug(
                            "Keyword '%s' found again outside TOC, setting extracting to True",
                            keyword,
                        )
                        break

            if extracting or in_toc:
                extracted_text.append(text)
            """
    # Let's try a new approach to extract text from the pdf
    # We'll use the PyMuPDF library to extract text from the pdf
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            text = page.get_text(sort=True)  # type: ignore
            if text:
                lines = text.split("\n")
                for line in lines[0:5]:
                    if "content" in line.lower().strip():
                        toc_passed = True
                        continue
                    if toc_passed and "one" in line.lower().strip():
                        extracting = True
                        break
            if extracting:
                extracted_text.append(text)

        cleaned_text = "\n".join(
            paragraph.strip() for paragraph in extracted_text if paragraph.strip()
        )
        logger.debug("Text extracted from pdf using PyMuPDF: %s", cleaned_text)
        return cleaned_text
        """
    # Trying out a new version using pymupdf's get_toc
    keywords = ["chapter 1", "chapter one", "1", "Intro", "Introduction"]
    extracting = False
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()
    for i in toc:


    except Exception as e:
        logger.warning(f"PyMuPDF failed to extract text: {e}")

    logger.info("Attempting to extract text using OCR")
    images = convert_from_path(pdf_path)
    ocr_text = ""
    for image in images:
        processed_image = preprocess_image(image)
        ocr_text += pytesseract.image_to_string(processed_image)
    return ocr_text.strip()


async def chunk_text(text: str, chars_per_chunk: int) -> list[str]:
    """
    Splits text into chunks while preserving sentence and clause boundaries.

    :param text: Input text to be chunked
    :param chars_per_chunk: Target size for each chunk
    :return: List of text chunks
    """
    # Split into clauses using the improved CLAUSE_BOUNDARIES pattern
    clauses = [c.strip() for c in re.split(CLAUSE_BOUNDARIES, text) if c and c.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for clause in clauses:
        clause_length = len(clause)
        logger.debug("Chunking text by clauses, currently on clause: %s", clause)

        # Handle clauses that are longer than chars_per_chunk
        if clause_length > chars_per_chunk:
            # If we have content in current_chunk, add it first
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = []
                current_length = 0

            # Split long clause at the last sentence boundary possible
            sentence_boundaries = r"(?<=[.!?])\s+"
            sentences = [
                s.strip() for s in re.split(sentence_boundaries, clause) if s.strip()
            ]

            for sentence in sentences:
                if len(sentence) > chars_per_chunk:
                    # If a single sentence is too long, split it into smaller parts
                    while sentence:
                        split_point = sentence.rfind(" ", 0, chars_per_chunk)
                        if split_point == -1:
                            split_point = chars_per_chunk
                        chunks.append(sentence[:split_point].strip())
                        sentence = sentence[split_point:].strip()
                else:
                    chunks.append(sentence)
            continue

        # Check if adding this clause would exceed the chunk size
        if current_length + clause_length + 1 > chars_per_chunk:
            # Save current chunk and start a new one
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [clause]
            current_length = clause_length
        else:
            # Add clause to current chunk
            current_chunk.append(clause)
            current_length += clause_length + 1  # +1 for space

    # Add the final chunk if there's anything left
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Remove any empty chunks and ensure proper spacing
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return chunks


async def openai_text_to_speech(
    text: str, output_path: Path, retries: int = 10, base_delay: int = 2
):
    for attempt in range(retries):
        try:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            async with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="shimmer",
                speed=1.2,
                input=text,
            ) as response:
                if response.status_code != 200:
                    raise ClientError(f"Invalid Response: {response.status_code}")
                await response.stream_to_file(output_path)
            return
        except (ClientError, Exception) as e:
            if attempt < retries - 1:
                logger.info(
                    f"Retrying after error: {e}. Attempt {attempt + 1} of {retries}"
                )
                delay = min(
                    base_delay * (2**attempt) + random.uniform(0, 1), 300
                )  # Max 5 minutes
                await asyncio.sleep(delay)
            else:
                logger.error("Failed after %d attempts", retries)
                raise


async def deepgram_text_to_speech(
    text: str, output_path: Path, retries: int = 15, base_delay: int = 60
):
    """
    The function `deepgram_text_to_speech` asynchronously converts text to speech using Deepgram API
    with retry mechanism and exponential backoff.

    :param text: The `text` parameter in the `deepgram_text_to_speech` function is a string that
    represents the text you want to convert into speech. This text will be used as input for the
    text-to-speech conversion process
    :type text: str
    :param output_path: The `output_path` parameter in the `deepgram_text_to_speech` function is a
    `Path` object that represents the file path where the text-to-speech output will be saved. You need
    to provide the full file path where you want the generated speech audio file to be saved on your
    :type output_path: Path
    :param retries: The `retries` parameter in the `deepgram_text_to_speech` function specifies the
    number of retry attempts that will be made in case the text-to-speech conversion fails before
    raising an error. In this function, if the initial text-to-speech conversion attempt fails, the
    function will retry, defaults to 15
    :type retries: int (optional)
    :param base_delay: The `base_delay` parameter in the `deepgram_text_to_speech` function represents
    the initial delay time in seconds before the first retry attempt is made. This delay time is then
    exponentially increased with each subsequent retry attempt using the formula `base_delay *
    (2**attempt) + random.uniform(), defaults to 15
    :type base_delay: int (optional)
    :return: The function `deepgram_text_to_speech` returns None explicitly. This is because there is no
    return value specified in the function. The function either successfully converts text to speech and
    logs the response, or it raises a RuntimeError if the conversion fails after multiple attempts.
    """

    # config: DeepgramClientOptions = DeepgramClientOptions(
    #        verbose=verboselogs.SPAM,
    # )
    config = ClientOptionsFromEnv()
    logger.debug("Running text-to-speech with deepgram for text: %s", text)
    deepgram = DeepgramClient(api_key="", config=config)
    options = SpeakOptions(
        model="aura-asteria-en",
    )

    for attempt in range(retries):
        try:
            response = await deepgram.speak.asyncrest.v("1").save(
                str(output_path), {"text": text}, options
            )
            logger.info("Text-to-speech conversion successful.")
            logger.debug(response.to_json(indent=4))
            return
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                # Calculate exponential backoff with jitter
                delay = base_delay + (2**attempt) + random.uniform(0, 1)
                logger.warning(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

    logger.error("Failed to convert text to speech after multiple attempts.")
    raise RuntimeError("Text-to-speech conversion failed after retries.")


async def process_chunk(chunk_text, chunk_index, base_name, output_dir, api):
    """
    Process the given text chunk by converting it to speech, skipping any existing audio files.
    """
    audio_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.mp3"

    # Check if the MP3 file already exists and is valid
    if audio_file.exists() and audio_file.stat().st_size > 1000:
        logger.info(
            "Skipping chunk %d, MP3 already exists: %s", chunk_index, audio_file
        )
        return audio_file

    logger.debug("Processing chunk %d: %s", chunk_index, chunk_text[:300])

    if len(chunk_text.strip()) == 0:
        logger.warning("Chunk %d is empty! Skipping conversion.", chunk_index)
        return None

    # Convert text directly to speech
    if api == "dg":
        await deepgram_text_to_speech(chunk_text, audio_file)
    elif api == "op":
        await openai_text_to_speech(chunk_text, audio_file)
    else:
        logger.error("Invalid API choice. Use 'deepgram' or 'openai'.")
        return None

    # Verify that MP3 file was created successfully
    if audio_file.exists() and audio_file.stat().st_size > 1000:
        logger.info(
            "Chunk %d successfully converted to speech: %s", chunk_index, audio_file
        )
        return audio_file
    else:
        logger.error(
            "Chunk %d conversion failed! File may be incomplete or corrupt: %s",
            chunk_index,
            audio_file,
        )
        return None


async def merge_audio_files(audio_files, output_path):
    """
    The function `merge_audio_files` merges multiple MP3 audio files into one main MP3 master file
    asynchronously.

    :param audio_files: A list of file paths to the individual audio files that you want to merge into
    one main audio file
    :param output_path: The `output_path` parameter in the `merge_audio_files` function is the file path
    where the merged audio files will be saved as a single main mp3 master file. This is the location
    where the combined audio from the input `audio_files` will be exported to after merging
    """
    logger.info("Merging all mp3 chunks into one main mp3 master file: %s", output_path)
    combined = AudioSegment.empty()
    for file in audio_files:
        if not os.path.exists(file):
            logger.error("Missing audio file: %s", file)
        else:
            logger.info("Audio file %s size: %d bytes", file, os.path.getsize(file))
        audio = await asyncio.to_thread(AudioSegment.from_mp3, file)
        combined = combined.append(audio, crossfade=200)
    await asyncio.to_thread(combined.export, output_path, format="mp3")


def cleanup(output_dir):
    """
    The `cleanup` function deletes specific files with certain extensions and removes the directory if
    it is empty.

    :param output_dir: output_dir is a directory path where files are stored. The cleanup function
    iterates through all files in the directory and deletes files with a ".txt" or ".mp3" extension that
    do not contain "merged" in their filename. If the directory becomes empty after deleting these
    files, the function also
    """
    for file in output_dir.glob("*"):
        try:
            if file.suffix in [".txt", ".mp3"] and "merged" not in file.stem:
                file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete {file}: {e}")
    try:
        if not any(output_dir.iterdir()):
            output_dir.rmdir()
    except Exception as e:
        logger.warning(f"Failed to delete directory {output_dir}: {e}")


async def main_async(pdf_file, api):
    """
    The `main_async` function processes a PDF file by extracting text, chunking it, converting chunks to
    audio files using an API, merging the audio files, and creating an audiobook.

    :param pdf_file: The `pdf_file` parameter in the `main_async` function is expected to be a string
    representing the path to a PDF file that you want to process and convert into an audiobook. This
    function uses various asynchronous operations to extract text from the PDF, chunk the text, process
    the chunks into audio
    :param api: The `api` parameter in the `main_async` function seems to be used for processing chunks
    of text into audio files. It is likely an API key or endpoint that is used for converting text to
    audio. This API could be a text-to-speech service or a similar tool that converts text data
    """
    pdf_path = Path(pdf_file)
    logger.debug(
        "Running main function with parameter of: %s which is of type: %s",
        pdf_path,
        type(pdf_path),
    )
    if not os.path.exists(pdf_path):
        logger.error("The specified PDF file does not exist.")
        sys.exit(1)
    logger.info("Welcome to our Audio Book Creator")
    logger.info("==================================")

    try:
        base_name = pdf_path.stem
        output_dir = pdf_path.parent / f"{base_name}_chunks"
        output_dir.mkdir(exist_ok=True)

        text = extract_text_from_pdf(pdf_path)
        logger.debug(
            "We've extracted text from your PDF: %s (Type: %s)", text, type(text)
        )
        logger.info("Now, we're going to chunk the text...")
        chars_per_chunk = 4096 if api == "op" else 2000
        chunks = await chunk_text(text, chars_per_chunk)
        audio_files = []
        tasks = [
            process_chunk(chunk, i, base_name, output_dir, api)
            for i, chunk in enumerate(chunks)
            if not (output_dir / f"{base_name}_chunk_{i + 1}.mp3").exists()
        ]

        if tasks:
            audio_files = []
            for task in tqdm(
                asyncio.as_completed(tasks), total=len(tasks), desc="Processing Chunks"
            ):
                result = await task
                if result:
                    audio_files.append(result)
        logger.info("Total text chunks created: %d", len(chunks))
        if len(chunks) == 0:
            logger.error("No text chunks created! Text extraction or chunking failed.")
            sys.exit(1)
    except Exception as e:
        logger.error("An error occurred while processing chunks: %s", e)
        sys.exit(1)

    try:
        merged_audio_file = output_dir / f"{base_name}_merged.mp3"
        await merge_audio_files(audio_files, merged_audio_file)
        logger.info(f"Audio book created successfully: {merged_audio_file}")
    except Exception as e:
        logger.error("Error during processing merge:%s", e)
    finally:
        cleanup(output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <choice of: deepgram or openai> <path_to_pdf>")
        sys.exit(1)
    # The code snippet is checking the value of the first command line argument (`sys.argv[1]`) and
    # setting the variable `api` to either "dg" if the argument is "deepgram" or "op" if the argument is
    # "openai".
    if sys.argv[1] == "deepgram":
        api = "dg"
    elif sys.argv[1] == "openai":
        api = "op"
    else:
        api = ""
try:
    asyncio.run(main_async(sys.argv[2], api))
except Exception as e:
    logger.error(f"Critical error: {e}")
    sys.exit(1)
