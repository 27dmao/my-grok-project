#!/usr/bin/env python3
"""Quick test to verify OPENAI_API_KEY is loaded from .env"""

from dotenv import load_dotenv
import os

load_dotenv()

xai_key = os.getenv("XAI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

print("Environment variables check:")
print(f"XAI_API_KEY: {'✓ Found' if xai_key else '✗ Not found'}")
print(f"OPENAI_API_KEY: {'✓ Found' if openai_key else '✗ Not found'}")

if openai_key:
    # Show first and last few characters for verification
    masked = f"{openai_key[:8]}...{openai_key[-4:]}" if len(openai_key) > 12 else "***"
    print(f"  Key preview: {masked}")
else:
    print("\nTo fix:")
    print("1. Make sure your .env file is in the project root: ~/Desktop/my-grok-project/.env")
    print("2. Add this line to .env:")
    print("   OPENAI_API_KEY=sk-your-actual-key-here")
    print("3. Make sure there are no spaces around the = sign")
    print("4. Restart the Flask app (python3 app.py)")

