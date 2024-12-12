import os
import sys
import aiohttp
import aiofiles
import asyncio
from pydub import AudioSegment


async def pdf_to_text(pdf_path):
    """Convert PDF to plaintext."""
    import PyPDF2  # Import here to keep the function modular.
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


async def split_text_into_files(text, base_filename):
    """Split text into multiple files, each containing 4096 characters."""
    chunk_size = 4096
    num_chunks = (len(text) // chunk_size) + (1 if len(text) % chunk_size != 0 else 0)

    async def write_chunk(chunk, filename):
        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(chunk)

    tasks = []
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        filename = f"{base_filename}_{i + 1}.txt"
        tasks.append(write_chunk(chunk, filename))
    await asyncio.gather(*tasks)


async def text_to_speech(file_path, openai_api_key):
    """Convert text file to MP3 using OpenAI API."""
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        text = await f.read()

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "input": text,
        "model": "tts-1",
        "voice": "shimmer",
        "speed": 1.2,
        "response_format": "mp3",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/audio/speech", headers=headers, json=data
        ) as response:
            if response.status == 200:
                mp3_filename = file_path.replace(".txt", ".mp3")
                async with aiofiles.open(mp3_filename, "wb") as mp3_file:
                    content = await response.read()
                    await mp3_file.write(content)
                print(f"Created MP3: {mp3_filename}")
                return mp3_filename
            else:
                print(f"Failed to convert {file_path}: {await response.text()}")
                return None


def merge_mp3_files(output_filename, mp3_files):
    """Merge multiple MP3 files into one large MP3 file."""
    combined = AudioSegment.empty()
    for mp3_file in mp3_files:
        audio_segment = AudioSegment.from_mp3(mp3_file)
        combined += audio_segment
    combined.export(output_filename, format="mp3")
    print(f"Merged MP3 file created: {output_filename}")


async def cleanup_files(base_filename):
    """Delete all chunked text files and individual MP3 files except the merged one."""
    for file in os.listdir():
        if file.startswith(base_filename):
            if file.endswith(".txt") or (
                file.endswith(".mp3") and not file.endswith("_merged.mp3")
            ):
                os.remove(file)
                print(f"Deleted: {file}")


async def main_async(pdf_path):
    if not os.path.exists(pdf_path):
        print("The specified PDF file does not exist.")
        sys.exit(1)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OpenAI API key is not set in the environment variables.")
        sys.exit(1)

    # Convert PDF to plaintext
    text = await pdf_to_text(pdf_path)
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    # Split text into multiple files of 4096 characters each
    await split_text_into_files(text, base_filename)

    # Convert each text file to MP3
    mp3_files = []
    num_chunks = (len(text) // 4096) + (1 if len(text) % 4096 != 0 else 0)
    tasks = []
    for i in range(num_chunks):
        text_file_path = f"{base_filename}_{i + 1}.txt"
        if os.path.exists(text_file_path):
            tasks.append(text_to_speech(text_file_path, openai_api_key))
    mp3_files = await asyncio.gather(*tasks)
    mp3_files = [mp3_file for mp3_file in mp3_files if mp3_file]  # Filter None values

    # Merge all MP3 files into one large MP3
    if mp3_files:
        merged_mp3_filename = f"{base_filename}_merged.mp3"
        merge_mp3_files(merged_mp3_filename, mp3_files)

        # Clean up individual MP3 files and text files
        await cleanup_files(base_filename)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_pdf>")
        sys.exit(1)

    asyncio.run(main_async(sys.argv[1]))
