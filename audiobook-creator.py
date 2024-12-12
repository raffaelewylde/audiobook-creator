#!/usr/bin/env python
import os
import asyncio
from pathlib import Path
from pydub import AudioSegment
from openai import AsyncOpenAI
from PyPDF2 import PdfReader
from aiohttp import ClientError

AsyncOpenAI.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages)
    return text

def chunk_text(text, chars_per_chunk=4096):
    """Split text into manageable chunks."""
    return [text[i:i + chars_per_chunk] for i in range(0, len(text), chars_per_chunk)]

async def text_to_speech_async(text, output_path, retries=3, delay=2):
    """Asynchronously convert text to speech with retries."""
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
                    raise ClientError(f"Invalid response: {response.status}")
                await response.stream_to_file(output_path)
            return
        except (ClientError, Exception) as e:
            if attempt < retries - 1:
                print(f"Retrying after error: {e}. Attempt {attempt + 1} of {retries}.")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                print(f"Failed after {retries} attempts.")
                raise

async def process_chunk(chunk_text, chunk_index, base_name, output_dir):
    """Process a single text chunk."""
    chunk_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.txt"
    audio_file = output_dir / f"{base_name}_chunk_{chunk_index + 1}.mp3"
    
    with open(chunk_file, "w", encoding="utf-8") as f:
        f.write(chunk_text)

    await text_to_speech_async(chunk_text, audio_file)
    return audio_file

async def merge_audio_files(audio_files, output_path):
    """Merge audio files into a single output."""
    combined = AudioSegment.empty()
    for file in audio_files:
        audio = await asyncio.to_thread(AudioSegment.from_mp3, file)
        combined += audio
    combined.export(output_path, format="mp3")

def cleanup(output_dir):
    """Delete all temporary files."""
    for file in output_dir.glob("*"):
        if file.suffix in [".txt", ".mp3"] and "merged" not in file.stem:
            file.unlink()
    if not any(output_dir.iterdir()):
        output_dir.rmdir()

async def main():
    print("Welcome to our Audio Book Creator")
    print("==================================")
    user_input = input("Enter the absolute file path of the PDF you'd like to convert: ")

    pdf_path = Path(user_input).resolve()
    if not pdf_path.is_file():
        print("Error: PDF file not found.")
        return
    
    text = extract_text_from_pdf(pdf_path)
    base_name = pdf_path.stem
    output_dir = pdf_path.parent / f"{base_name}_chunks"
    output_dir.mkdir(exist_ok=True)

    tasks = [
        process_chunk(chunk, i, base_name, output_dir)
        for i, chunk in enumerate(chunk_text(text))
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
    asyncio.run(main())
