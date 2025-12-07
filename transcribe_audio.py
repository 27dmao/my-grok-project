"""
Audio transcription helper script.

This is a placeholder that shows how to integrate transcription.
Replace the transcribe_audio() function with your actual transcription service.
"""

import os
import sys
from pathlib import Path


def transcribe_audio_openai(file_path):
    """
    Transcribe audio using OpenAI's Whisper API.
    
    Requires: pip install openai
    Requires: OPENAI_API_KEY in .env
    """
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        return transcript
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        sys.exit(1)
    except Exception as e:
        print(f"Error transcribing with OpenAI: {e}")
        sys.exit(1)


def transcribe_audio_whisper_local(file_path):
    """
    Transcribe audio using local Whisper model.
    
    Requires: pip install openai-whisper
    """
    try:
        import whisper
        
        model = whisper.load_model("base")  # or "tiny", "small", "medium", "large"
        result = model.transcribe(file_path)
        
        return result["text"]
    except ImportError:
        print("Error: whisper package not installed. Run: pip install openai-whisper")
        sys.exit(1)
    except Exception as e:
        print(f"Error transcribing with local Whisper: {e}")
        sys.exit(1)


def main():
    """
    Transcribe an audio file and save to transcript.txt
    
    Usage:
        python3 transcribe_audio.py audio.mp3 [--method openai|whisper]
    """
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe_audio.py <audio_file> [--method openai|whisper]")
        print("\nMethods:")
        print("  openai  - Use OpenAI Whisper API (requires OPENAI_API_KEY)")
        print("  whisper - Use local Whisper model (requires openai-whisper package)")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    method = "openai"
    
    if len(sys.argv) > 2 and sys.argv[2] == "--method":
        if len(sys.argv) > 3:
            method = sys.argv[3]
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
    
    print(f"Transcribing {audio_path} using {method}...")
    
    if method == "openai":
        transcript = transcribe_audio_openai(audio_path)
    elif method == "whisper":
        transcript = transcribe_audio_whisper_local(audio_path)
    else:
        print(f"Error: Unknown method '{method}'. Use 'openai' or 'whisper'.")
        sys.exit(1)
    
    # Save transcript
    output_file = "transcript.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    print(f"\nTranscript saved to {output_file}")
    print(f"\nFirst 500 characters:")
    print(transcript[:500] + "..." if len(transcript) > 500 else transcript)


if __name__ == "__main__":
    main()

