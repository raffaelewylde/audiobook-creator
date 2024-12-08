# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pydub",
#     "pypdf2",
#     "requests",
# ]
# ///

import os
import sys
import PyPDF2
import requests
from pydub import AudioSegment

def pdf_to_text(pdf_path):
    """Convert PDF to plaintext."""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_text_into_files(text, base_filename):
    """Split text into multiple files, each containing 4096 characters."""
    chunk_size = 4096
    num_chunks = (len(text) // chunk_size) + (1 if len(text) % chunk_size != 0 else 0)
    
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        with open(f"{base_filename}_{i + 1}.txt", "w", encoding="utf-8") as f:
            f.write(chunk)

def text_to_speech(file_path):
    """Convert text file to MP3 using OpenAI API."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        'Authorization': f'Bearer {openai_api_key}',
        'Content-Type': 'application/json'
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    data = {
        "input": text,
        "voice": "shimmer",  # Specify the voice you want
        "model": "tts-1",
        "speed": 1.2,
        "response_format": "mp3"
    }

    response = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=data)

    if response.status_code == 200:
        mp3_filename = file_path.replace('.txt', '.mp3')
        with open(mp3_filename, 'wb') as mp3_file:
            mp3_file.write(response.content)
        print(f"Created MP3: {mp3_filename}")
        return mp3_filename
    else:
        print(f"Failed to convert {file_path}: {response.text}")
        return None

def merge_mp3_files(output_filename, mp3_files):
    """Merge multiple MP3 files into one large MP3 file."""
    combined = AudioSegment.empty()
    for mp3_file in mp3_files:
        audio_segment = AudioSegment.from_mp3(mp3_file)
        combined += audio_segment
    combined.export(output_filename, format="mp3")
    print(f"Merged MP3 file created: {output_filename}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print("The specified PDF file does not exist.")
        sys.exit(1)

    # Convert PDF to plaintext
    text = pdf_to_text(pdf_path)
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    # Split text into multiple files of 4096 characters each
    split_text_into_files(text, base_filename)

    mp3_files = []
    # Convert each text file to MP3
    for i in range((len(text) // 4096) + 1):
        text_file_path = f"{base_filename}_{i + 1}.txt"
        if os.path.exists(text_file_path):
            mp3_file = text_to_speech(text_file_path)
            if mp3_file:
                mp3_files.append(mp3_file)

    # Merge all MP3 files into one large MP3
    if mp3_files:
        merge_mp3_files(f"{base_filename}_merged.mp3", mp3_files)

if __name__ == "__main__":
    main()


