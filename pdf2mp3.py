#!/usr/bin/env python

# Welcome to a python script designed to create an audiobook from a pdf file

import os
import argparse
import asyncio
import PyPDF2
from pathlib import Path
from pydub import AudioSegment
import openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def chunk_text(text, words_per_chunk=1000):
    """split text into chunks of for better uploading speed."""
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        yield " ".join(words[i : i + words_per_chunk])


async def text_to_speech_async(text, output_path):
    """Asynchronously convert text to speech with openai"""
        client - OpenAI()
        response = await client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=text
        )

    response.stream_to_file(output_path)

async def process_chunk(chunk_text, chunk_index, base_name, output_dir):
    """Process a single chunk: save text to file, convert to speech, save to mp3."""
    chunk_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.txt"
    audio_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.mp3"

    # Save chunk to text file
    with open(chunk_file, "w") as f:
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

async def main():
    parser = argparse.ArgumentParser(description="Convert a PDF book into MP3")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_file)
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
        tasks.append(process_chunk(chunk, i, base_name, output_dir))

    audio_files = await asyncio.gather(*tasks)

    # Optionally merge audio files into one
    merged_audio_file = pdf_path.parent / f"{base_name}_merged.mp3"
    await merge_audio_files(audio_files, merged_audio_file)

if __name__ == "__main__":
    asyncio.run(main())
