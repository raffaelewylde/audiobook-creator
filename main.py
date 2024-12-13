import logging
from logging.handlers import RotatingFileHandler
import random
import os
import sys
import aiofiles
import asyncio
from pathlib import Path
from deepgram.utils import verboselogs
from pydub import AudioSegment
from PyPDF2 import PdfReader

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    ClientOptionsFromEnv,
    SpeakOptions,
)

verboselogs.install()

def setup_logging():
    """
    The `setup_logging` function configures logging for an application, setting
    up both console and file handlers with specific levels and formatting.
    :return: The `setup_logging` function returns a logger object that is
    configured to log messages to both the console and a file named "app.log".
    The logger is set to log messages at the INFO level for the console handler
    and at the DEBUG level for the file handler. The logger includes a
    formatter that specifies the format of the log messages.
    """
    logger = logging.getLogger(__name__)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        "app.log", maxBytes=10000000, backupCount=5
    )  # noqa: F821
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


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    logger.info("Converting your pdf, %s, to plain text", pdf_path)
    filetype = type(pdf_path)
    logger.debug("Type for pdf is %s", filetype)
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        logger.debug("Opened pdf for reading as type: %s becoming type: %s", type(file), type(reader))
        text = "".join(page.extract_text() for page in reader.pages)
    logger.debug("Text extracted from pdf: %s", text)
    return text


async def chunk_text(text: str, chars_per_chunk: int = 2000) -> list[str]:
    """Split text into manageable chunks."""
    chunked_text = [text[i : i + chars_per_chunk] for i in range(0, len(text), chars_per_chunk)]
    logger.debug("Chunked text type: %s, chunked text: %s", type(chunked_text), chunked_text)
    return chunked_text


async def text_to_speech(
    text: str, output_path: Path, retries: int = 5, base_delay: int = 2
):
    """
    Convert text to speech using Deepgram, with exponential backoff for retries.

    :param text: The text to convert to speech.
    :param output_path: The output path for the audio file.
    :param retries: The maximum number of retries.
    :param base_delay: The initial delay for exponential backoff in seconds.
    """
    #config: DeepgramClientOptions = DeepgramClientOptions(
    #        verbose=verboselogs.SPAM,
    #)
    config=ClientOptionsFromEnv()
    logger.debug("DeepgramClientOptions: %s", config)
    deepgram = DeepgramClient(api_key="", config=config)
    options = SpeakOptions(
        model="aura-angus-en",
    )

    for attempt in range(retries):
        try:
            response = await deepgram.speak.asyncrest.v("1").save(output_path, text, options)
            logger.info("Text-to-speech conversion successful.")
            logger.info(response.to_json(indent=4))
            return
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                # Calculate exponential backoff with jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

    logger.error("Failed to convert text to speech after multiple attempts.")
    raise RuntimeError("Text-to-speech conversion failed after retries.")


async def process_chunk(chunk_text, chunk_index, base_name, output_dir):
    """Process a single text chunk."""
    logger.info("Processing your chunked text now...")
    # chunk_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.txt"
    audio_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.mp3"
    logger.debug("Audio file: %s", audio_file)
    logger.debug("parameters that were passed to process_chunk: %s, %s, %s, %s", chunk_text, chunk_index, base_name, output_dir)

    # async with aiofiles.open(chunk_file, "w", encoding="utf-8") as f:
    #   logger.debug("writing chunk text to file: %s", chunk_file)
    #    await f.write(chunk_text)
    
    # async with aiofiles.open(chunk_file, "r", encoding="utf-8") as f:
    #   logger.debug("reading text from file: %s", chunk_file)
    #    text_to_convert = await f.read(chunk_file)
    await text_to_speech(chunk_text, audio_file)
    return audio_file


async def merge_audio_files(audio_files, output_path):
    """Merge audio files into a single output."""
    logger.info("Merging all mp3 chunks into one main mp3 master file: %s", output_path)
    combined = AudioSegment.empty()
    for file in audio_files:
        audio = await asyncio.to_thread(AudioSegment.from_mp3, file)
        combined += audio
    await asyncio.to_thread(combined.export, output_path, format="mp3")


def cleanup(output_dir):
    """Delete all temporary files."""
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


async def main_async(pdf_path):
    logger.debug("Running main function with parameter of: %s which is of type: %s", pdf_path, type(pdf_path))
    if not os.path.exists(pdf_path):
        print("The specified PDF file does not exist.")
        sys.exit(1)
    logger.info("Welcome to our Audio Book Creator")
    logger.info("==================================")

    text = extract_text_from_pdf(pdf_path)
    logger.debug("We've extracted text from your PDF: %s", text, type(text))
    base_name = pdf_path.stem
    output_dir = pdf_path.parent / f"{base_name}_chunks"
    output_dir.mkdir(exist_ok=True)

    chunks = await chunk_text(text)
    tasks = [
        process_chunk(chunk, i, base_name, output_dir) for i, chunk in enumerate(chunks)
    ]

    try:
        audio_files = await asyncio.gather(*tasks)
        merged_audio_file = pdf_path.parent / f"{base_name}_merged.mp3"
        await merge_audio_files(audio_files, merged_audio_file)
        print(f"Audio book created successfully: {merged_audio_file}")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        cleanup(output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_pdf>")
        sys.exit(1)

    asyncio.run(main_async(sys.argv[1]))
