import os
import json
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY not found in environment. Check your .env file.")

XAI_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-0709"

EMO_PROMPT = """
You are an emotional mapping assistant.

You receive a single conversation transcript. Your job is to:
1) Map likely emotional states over the course of the conversation.
2) Identify key shifts and triggers.
3) Output a machine-usable JSON structure.

CONSTRAINTS:
- Do NOT diagnose mental health conditions.
- Do NOT claim to know the "true" internal state, only inferred emotions based on language.
- Do NOT talk about trauma as a fact; you may mention "possible emotional wounding"
  only as a gentle hypothesis, not as a label.

OUTPUT FORMAT:
Return STRICTLY valid JSON with:

{
  "timeline": [
    {
      "segment_id": 1,
      "text_snippet": "short snippet...",
      "approx_position": "start|middle|end",
      "speaker": "A/B/unknown",
      "inferred_emotions": ["anxious", "hopeful"],
      "intensity": "low|medium|high",
      "notes": "short natural language note"
    },
    ...
  ],
  "global_summary": {
    "baseline_tone": "e.g. generally warm but slightly anxious",
    "main_emotions": ["emotion1", "emotion2"],
    "key_triggers": [
      "topic or moment that seems to shift emotional tone"
    ],
    "regulation_style": "how they seem to manage difficult feelings, in plain language",
    "reflection_prompts": [
      "journal-style questions to help the person gain awareness of these patterns"
    ]
  }
}

Focus on clarity and usefulness, not clinical language.
""".strip()


def call_grok_for_emotions(transcript: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": EMO_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "stream": False,
    }

    resp = requests.post(XAI_URL, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()
    raw_content = data["choices"][0]["message"]["content"]

    start = raw_content.find("{")
    end = raw_content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model did not return JSON-like content.")

    return json.loads(raw_content[start : end + 1])


def main():
    parser = argparse.ArgumentParser(
        description="Map emotions over a transcript for HumanIntuition.ai."
    )
    parser.add_argument("transcript", help="Path to a transcript text file.")
    parser.add_argument(
        "--output",
        type=str,
        default="emotional_map.json",
        help="Output JSON file.",
    )
    args = parser.parse_args()

    with open(args.transcript, "r", encoding="utf-8") as f:
        transcript = f.read()

    emo_map = call_grok_for_emotions(transcript)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(emo_map, f, indent=2, ensure_ascii=False)

    print(f"Saved emotional map to {args.output}")


if __name__ == "__main__":
    main()
