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

CLAUSE_BOUNDARIES = r"\.\?|!|;|, (?:and|but|or|nor|for|yet|so)"
AsyncOpenAI.api_key = os.getenv("OPENAI_API_KEY")


def setup_logging():
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
    image = image.convert("L")
    image = image.point(lambda x: 0 if x < 128 else 255)
    return image


def extract_text_from_pdf(pdf_path):
    logger.info("Converting your pdf, %s, to plain text", pdf_path)
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages)
    return text


async def pre_scan_output_dir(output_dir, base_name):
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
    clauses = re.split(CLAUSE_BOUNDARIES, text)
    clauses = [clause.strip() for clause in clauses if clause and clause.strip()]

    chunks = []
    current_chunk = ""

    for clause in clauses:
        while len(clause) > chars_per_chunk:
            chunks.append(clause[:chars_per_chunk])
            clause = clause[chars_per_chunk:]

        if len(current_chunk) + len(clause) + 1 <= chars_per_chunk:
            current_chunk += clause + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = clause + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


async def save_chunks_to_file(chunks, output_dir, base_name):
    chunk_file_path = output_dir / f"{base_name}_chunks.txt"
    async with aiofiles.open(chunk_file_path, "w", encoding="utf-8") as f:
        await f.write("\n\n".join(chunks))


async def load_chunks_from_file(output_dir, base_name):
    chunk_file_path = output_dir / f"{base_name}_chunks.txt"
    if chunk_file_path.exists():
        async with aiofiles.open(chunk_file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return content.split("\n\n")
    return None


async def openai_text_to_speech(
    text: str, output_path: Path, retries: int = 5, base_delay: int = 2
):
    base_delay = min(base_delay, 10)
    retries = min(retries, 5)

    for attempt in range(retries):
        try:
            client = AsyncOpenAI()
            async with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="shimmer",
                speed=1.2,
                input=text,
            ) as response:
                try:
                    await response.stream_to_file(output_path)
                except Exception as e:
                    logger.error(
                        "Failed to stream audio with openai api to file: %s", e
                    )
        except (ClientError, Exception) as e:
            if attempt < retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                logger.error("Failed after %d attempts", retries)
                raise


async def deepgram_text_to_speech(
    text: str, output_path: Path, retries: int = 5, base_delay: int = 10
):
    base_delay = min(base_delay, 10)
    retries = min(retries, 5)

    config = ClientOptionsFromEnv()
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
            return
        except Exception as e:
            if attempt < retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                logger.error("Failed after %d attempts", retries)
                raise


async def process_chunk(chunk_text, chunk_index, base_name, output_dir, api):
    chunk_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.txt"
    audio_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.mp3"

    try:
        if not os.path.exists(chunk_file):
            async with aiofiles.open(chunk_file, "w", encoding="utf-8") as f:
                await f.write(chunk_text)

        if not os.path.exists(audio_file):
            async with aiofiles.open(chunk_file, "r", encoding="utf-8") as f:
                text_to_convert = await f.read()
            if api == "dg":
                await deepgram_text_to_speech(text_to_convert, audio_file)
            elif api == "op":
                await openai_text_to_speech(text_to_convert, audio_file)
            else:
                logger.error("Invalid API choice.")
                return None
        return audio_file

    except Exception as e:
        logger.error("An error occurred while processing chunks: %s", e)


async def merge_audio_files(audio_files, output_path):
    logger.info("Merging all mp3 chunks into one master file: %s", output_path)
    combined = AudioSegment.empty()
    for file in audio_files:
        audio = await asyncio.to_thread(AudioSegment.from_mp3, file)
        combined += audio
    await asyncio.to_thread(combined.export, output_path, format="mp3")


def cleanup(output_dir):
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
    pdf_path = Path(pdf_file)
    if not os.path.exists(pdf_path):
        logger.error("The specified PDF file does not exist.")
        sys.exit(1)

    try:
        base_name = pdf_path.stem
        output_dir = pdf_path.parent / f"{base_name}_chunks"
        output_dir.mkdir(exist_ok=True)

        processed_indices = await pre_scan_output_dir(output_dir, base_name)

        chunks = await load_chunks_from_file(output_dir, base_name)
        if chunks is None:
            text = extract_text_from_pdf(pdf_path)
            chars_per_chunk = 4096 if api == "op" else 2000
            chunks = await chunk_text(text, chars_per_chunk)
            await save_chunks_to_file(chunks, output_dir, base_name)

        filtered_chunks = [
            (i, chunk)
            for i, chunk in enumerate(chunks)
            if i + 1 not in processed_indices
        ]

        audio_files = []
        for batch_start in range(0, len(filtered_chunks), 3):
            batch_end = min(batch_start + 3, len(filtered_chunks))
            batch = filtered_chunks[batch_start:batch_end]

            tasks = [
                process_chunk(chunk, i, base_name, output_dir, api)
                for i, chunk in batch
            ]
            if not tasks:
                continue
            batch_audio_files = await asyncio.gather(*tasks)
            audio_files.extend(filter(None, batch_audio_files))

        merged_audio_file = output_dir / f"{base_name}_merged.mp3"
        await merge_audio_files(audio_files, merged_audio_file)
        logger.info(f"Audiobook created successfully: {merged_audio_file}")

    except Exception as e:
        logger.error("An error occurred: %s", e)
    finally:
        cleanup(output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <choice of: deepgram or openai> <path_to_pdf>")
        sys.exit(1)
    api = "dg" if sys.argv[1] == "deepgram" else "op" if sys.argv[1] == "openai" else ""
    if not api:
        print("Invalid API choice. Use 'deepgram' or 'openai'.")
        sys.exit(1)
    asyncio.run(main_async(sys.argv[2], api))
