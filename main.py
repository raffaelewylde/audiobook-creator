import logging
import re
from logging.handlers import RotatingFileHandler
import random
import os
import sys
import aiofiles
import asyncio
from pathlib import Path
from pydub import AudioSegment
from PyPDF2 import PdfReader
from openai import AsyncOpenAI
from aiohttp import ClientError
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from tqdm.asyncio import tqdm

from deepgram import (
    DeepgramClient,
    ClientOptionsFromEnv,
    SpeakOptions,
)

#CLAUSE_BOUNDARIES = r"\.|\?|!|;|, (?:and|but|or|nor|for|yet|so)"
CLAUSE_BOUNDARIES = r"(?<=[.?!;])\s+|(?<!\w)\n"

AsyncOpenAI.api_key = os.getenv("OPENAI_API_KEY")


def setup_logging():
    """
    The `setup_logging` function configures logging to output messages to both the console and a
    rotating file.
    :return: A logger object with both a console handler and a file handler set up for logging.
    """
    logger = logging.getLogger(__name__)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler("app.log", maxBytes=10000000, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s -  %(levelname)s - %(message)s - Line: %(lineno)d"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.point(lambda x: 0 if x < 128 else 255)  # Binarize
    return image

def extract_text_from_pdf(pdf_path):
    """
    The function `extract_text_from_pdf` reads a PDF file and extracts its text content.

    :param pdf_path: The `pdf_path` parameter in the `extract_text_from_pdf` function is a string that
    represents the file path to the PDF file from which you want to extract text
    :return: The function `extract_text_from_pdf` returns the extracted text from the PDF file located
    at the `pdf_path` provided as input.
    """
    logger.info("Converting your pdf, %s, to plain text", pdf_path)
    print("Converting your pdf, %s, to plain text", pdf_path)
    filetype = type(pdf_path)
    logger.debug("Type for pdf is %s", filetype)
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            paragraphs = text.split('\n')
            cleaned_text = '\n'.join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
            logger.info("Text extracted from pdf using PyPDF2: %s", cleaned_text)
            print(f"Text extracted from pdf using PyPDF2: {cleaned_text}")
            return cleaned_text
    except Exception as e:
        logger.warning(f"PyPDF2 failed to extract text: {e}")

    logger.info("Attempting to extract text using OCR")
    print("Attempting to extract text using OCR")
    images = convert_from_path(pdf_path)
    ocr_text = ""
    for image in images:
        processed_image = preprocess_image(image)
        ocr_text += pytesseract.image_to_string(processed_image)
    return ocr_text.strip()

async def pre_scan_output_dir(output_dir, base_name):
    """
    Pre-scans the output directory to identify already processed chunk indices based on existing files.

    :param output_dir: Directory where chunk text and audio files are stored.
    :param base_name: Base name of the chunk files.
    :return: A set of indices representing already processed chunks.
    """
    processed_indices = set()
    chunk_files = await asyncio.to_thread(
        list, output_dir.glob(f"{base_name}_chunk_*.txt")
    )
    for file in chunk_files:
        match = re.search(rf"{base_name}_chunk_(\d+)", file.stem)
        if match:
            processed_indices.add(int(match.group(1)))
    return processed_indices


async def chunk_text(text: str, chars_per_chunk: int) -> list[str]:
    """
    The `chunk_text` function splits a given text into chunks of specified length while preserving
    clause boundaries.

    :param text: The `chunk_text` function you provided is designed to split a given text into chunks
    based on a specified number of characters per chunk. It first splits the text into clauses using a
    regular expression pattern `CLAUSE_BOUNDARIES`, then processes these clauses to create chunks of
    text that do not exceed the
    :type text: str
    :param chars_per_chunk: The `chars_per_chunk` parameter specifies the maximum number of characters
    allowed in each chunk of text when splitting the input text into chunks. This parameter helps
    control the size of each chunk to ensure that they are within a manageable length for processing or
    displaying purposes, defaults to 2000
    :type chars_per_chunk: int (optional)
    :return: The function `chunk_text` returns a list of strings, where each string represents a chunk
    of text that has been split based on the specified number of characters per chunk.
    """
    clauses = re.split(CLAUSE_BOUNDARIES, text)
    clauses = [
        clause.strip() for clause in clauses if clause and clause.strip()
    ]  # Clean up empty or whitespace-only clauses

    chunks = []
    current_chunk = ""

    for clause in clauses:
        # Check if the clause itself is too long
        while len(clause) > chars_per_chunk:
            # Add as much of the clause as possible to a new chunk
            chunks.append(clause[:chars_per_chunk])
            clause = clause[chars_per_chunk:]  # Trim the part that's been added

        # Add the clause if it fits within the current chunk
        if len(current_chunk) + len(clause) + 1 <= chars_per_chunk:
            current_chunk += clause + " "
        else:
            # Add the current chunk to the list and start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = clause + " "

    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


async def openai_text_to_speech(
    text: str, output_path: Path, retries: int = 10, base_delay: int = 2
):
    """
    This Python async function utilizes the OpenAI API to convert text to speech and save the output to
    a specified file path, with retry and error handling mechanisms in place.

    :param text: The `text` parameter in the `openai_text_to_speech` function is a string that
    represents the text you want to convert to speech. This text will be used as input for the
    text-to-speech conversion process
    :type text: str
    :param output_path: The `output_path` parameter in the `openai_text_to_speech` function is the path
    where the generated audio file will be saved. It should be a `Path` object representing the location
    where you want to save the audio file. You can provide the full path including the file name and
    :type output_path: Path
    :param retries: The `retries` parameter in the `openai_text_to_speech` function specifies the number
    of times the function will attempt to perform the text-to-speech conversion in case of errors or
    failures. If an error occurs during the initial attempt, the function will retry the operation up to
    the specified, defaults to 10
    :type retries: int (optional)
    :param base_delay: The `delay` parameter in the `openai_text_to_speech` function represents the initial
    time delay in seconds between retry attempts when an error occurs during the text-to-speech
    conversion process. This delay is doubled after each unsuccessful attempt to give the system some
    time before retrying, defaults to 2
    :type base_delay: int (optional)
    :return: The `openai_text_to_speech` function returns `None` if the text-to-speech conversion is
    successful and the audio file is saved to the specified `output_path`. If there are errors during
    the process and all retry attempts are exhausted, an exception will be raised.
    """
    for attempt in range(retries):
        try:
            client = AsyncOpenAI()
            async with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="shimmer",
                speed=1.2,
                input=text,
            ) as response:
                if response.status != 200:
                    raise ClientError(f"Invalid Response: {response.status}")
                await response.stream_to_file(output_path)
            return
        except (ClientError, Exception) as e:
            if attempt < retries - 1:
                logger.info(
                    f"Retrying after error: {e}. Attempt {attempt + 1} of {retries}"
                )
                print(f"Retrying after error: {e}. Attempt {attempt + 1} of {retries}.")
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                logger.error("Failed after %d attempts", retries)
                raise


async def deepgram_text_to_speech(
    text: str, output_path: Path, retries: int = 15, base_delay: int = 15
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
    logger.debug("DeepgramClientOptions: %s", config)
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
            print("Text-to-speech conversion successful.")
            logger.info(response.to_json(indent=4))
            print(response.to_json(indent=4))
            return
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                # Calculate exponential backoff with jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

    logger.error("Failed to convert text to speech after multiple attempts.")
    raise RuntimeError("Text-to-speech conversion failed after retries.")


async def process_chunk(chunk_text, chunk_index, base_name, output_dir, api):
    """
    This Python async function processes chunked text by writing it to a file and converting it to
    speech using either Deepgram or OpenAI APIs.

    :param chunk_text: The `chunk_text` parameter in the `process_chunk` function is the text content of
    a chunk that needs to be processed. This text will be written to a file and then converted to speech
    using a specified API (either "dg" for Deepgram or "op" for OpenAI).
    :param chunk_index: The `chunk_index` parameter in the `process_chunk` function represents the index
    of the current chunk being processed. It is used to uniquely identify each chunk and is incremented
    for each new chunk processed
    :param base_name: The `base_name` parameter in the `process_chunk` function is a string that
    represents the base name for the output files that will be generated during the processing of the
    chunked text. It is used to construct the names of the chunk text file and the audio file for each
    chunk
    :param output_dir: The `output_dir` parameter in the `process_chunk` function represents the
    directory where the output files will be saved. It is the directory path where the text and audio
    files for each chunk will be stored
    :param api: The `api` parameter in the `process_chunk` function is used to specify which Text to
    Speech API to use for converting the text chunk into an audio file. The function checks the value of
    `api` to determine whether to use the Deepgram (`dg`) or OpenAI (`op`) Text
    :return: The function `process_chunk` is returning the path to the audio file that was generated for
    the chunk of text processed. If the audio file already exists for the chunk, it will log a message
    and return without processing the chunk further.
    """
    chunk_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.txt"
    audio_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.mp3"
    logger.debug("Audio file: %s", audio_file)
    logger.debug(
        "parameters that were passed to process_chunk: %s, %s, %s, %s",
        chunk_text,
        chunk_index,
        base_name,
        output_dir,
    )

    if not os.path.exists(chunk_file):
        async with aiofiles.open(chunk_file, "w", encoding="utf-8") as f:
            logger.debug("writing chunk text to file: %s", chunk_file)
            await f.write(chunk_text)

    if not os.path.exists(audio_file):
        async with aiofiles.open(chunk_file, "r", encoding="utf-8") as f:
            logger.debug("reading text from file: %s", chunk_file)
            text_to_convert = await f.read()
        if api == "dg":
            await deepgram_text_to_speech(text_to_convert, audio_file)
            return audio_file
        elif api == "op":
            await openai_text_to_speech(text_to_convert, audio_file)
            return audio_file
        else:
            logger.error(
                "It seems your choice of APIs to use for Text to Speech is misconfigured. It Should be either openai or deepgram and passed as a commandline option before the path to your pdf file."
            )
            print(
                "It seems your choice of APIs to use for Text to Speech is misconfigured. It Should be either openai or deepgram and passed as a commandline option before the path to your pdf file."
            )
        return audio_file
    else:
        logger.info(
            f"Files already exist for this chunk, continuing on. The chunk was: {chunk_text}"
        )
        print(
            f"Files already exist for this chunk, continuing on. The chunk was: {chunk_text}"
        )
        return


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
    print(f"Merging all mp3 chunks into one main mp3 master file: {output_path}")
    combined = AudioSegment.empty()
    for file in audio_files:
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
        print("The specified PDF file does not exist.")
        sys.exit(1)
    logger.info("Welcome to our Audio Book Creator")
    print("Welcome to our Audio Book Creator")
    logger.info("==================================")
    print("==================================")

    try:
        base_name = pdf_path.stem
        output_dir = pdf_path.parent / f"{base_name}_chunks"
        output_dir.mkdir(exist_ok=True)

        logger.info("Pre-scanning for already processed chunks...")
        print("Pre-scanning for already processed chunks...")
        processed_indices = await pre_scan_output_dir(output_dir, base_name)
        logger.info("Found %d already processed chunks.", len(processed_indices))
        print(f"Found {len(processed_indices)} already processed chunks")

        text = extract_text_from_pdf(pdf_path)
        logger.debug("We've extracted text from your PDF: %s", text, type(text))

        logger.info("Now, we're going to chunk the text...")
        print("Now, we're going to chunk the text...")

        chars_per_chunk = 4096 if api == "op" else 2000
        chunks = await chunk_text(text, chars_per_chunk)

        audio_files = []
        tasks = [
            process_chunk(chunk, i, base_name, output_dir, api)
            for i, chunk in enumerate(chunks)
            if i + 1 not in processed_indices
        ]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Chunks"):
            result = await task
            if result:
                audio_files.append(result)
    except Exception as e:
        logger.error("An error occurred while processing chunks: %s", e)
        print(f"An error occurred while processing chunks: {e}")
        sys.exit(1)

    try:
        merged_audio_file = output_dir / f"{base_name}_merged.mp3"
        await merge_audio_files(audio_files, merged_audio_file)
        print(f"Audio book created successfully: {merged_audio_file}")
    except Exception as e:
        logger.error("Error during processing merge:%s", e)
        print(f"Error during processing merge: {e}")
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
    asyncio.run(main_async(sys.argv[2], api))
