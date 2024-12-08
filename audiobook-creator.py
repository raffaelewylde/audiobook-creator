#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai",
#     "pydub",
#     "pypdf2",
# ]
# ///


import os
import asyncio
import PyPDF2
from pathlib import Path
from pydub import AudioSegment
from openai import AsyncOpenAI

AsyncOpenAI.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def chunk_text(text, chars_per_chunk=4096):
    """split text into chunks of for better uploading speed."""
    for i in range(0, len(text), chars_per_chunk):
        yield text[i : i + chars_per_chunk]

async def text_to_speech_async(text, output_path):
    """Asynchronously convert text to speech with openai"""
    client = AsyncOpenAI()
    async with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="shimmer",
        input=text
    ) as response:
        await response.stream_to_file(output_path)

async def process_chunk(chunk_text, chunk_index, base_name, output_dir):
    """Process a single chunk: save text to file, convert to speech, save to mp3."""
    chunk_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.txt"
    audio_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.mp3"

    # Save chunk to text file
    with open(chunk_file, "w", encoding="utf-8") as f:
        f.write(chunk_text)

    # Generate audio from text
    await text_to_speech_async(chunk_text, audio_file)

    return audio_file


async def merge_audio_files(audio_files, output_path):
    """Merge multiple audio files into one."""
    combined = AudioSegment.empty()
    for file in audio_files:
        audio = await asyncio.to_thread(AudioSegment.from_mp3, file)
        combined += audio
    combined.export(output_path, format="mp3")


def cleanup():
    """Remove temporary files."""
    output_dir = Path(pdf_path).parent / f"{pdf_path.stem}_chunks"
    output_dir.rmdir()
async def main():
    print("Welcome to our Audio Book Creator")
    print("==================================")
    user_input = input("Where is the pdf located you'd like to generate an audiobook for? Provide absolute file path")

    pdf_file = Path(user_input)
    pdf_path = Path(pdf_file).resolve()
    if not pdf_path.is_file():
        print("Error: Pdf file not found.")
        return
    text = extract_text_from_pdf(pdf_path)
    base_name = pdf_path.stem
    output_dir = pdf_path.parent / f"{base_name}_chunks"
    output_dir.mkdir(exist_ok=True)

    # Process text chunks in parallel
    tasks = []
    for i, chunk in enumerate(chunk_text(text)):
        if len(chunk) > 4096:
            print(f"Chunk {i + 1} exceeds the character limit!")
            continue
        tasks.append(process_chunk(chunk, i, base_name, output_dir))

    audio_files = await asyncio.gather(*tasks)

    # Optionally merge audio files into one
    merged_audio_file = pdf_path.parent / f"{base_name}_merged.mp3"
    await merge_audio_files(audio_files, merged_audio_file)

if __name__ == "__main__":
    asyncio.run(main())

