import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY not found in environment. Check your .env file.")

XAI_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-0709"  # or your preferred Grok model

PROFILE_PROMPT = """
You are building a deep, non-clinical behavioral profile for a single person
based on several transcripts of their real conversations.

IMPORTANT CONSTRAINTS:
- You are NOT a therapist and you must NOT make mental health or medical diagnoses.
- You must NOT label trauma, attachment style, or personality types as facts.
- You may use concepts like "shadow work", "attachment patterns", "Enneagram flavor"
  as loose interpretive lenses ONLY, never as clinical truth.
- You must NOT claim to detect lies, truthfulness, or deception.

YOUR GOAL:
Create a structured JSON object that describes this person's:

1. core_narratives: recurring beliefs and stories (e.g. "I must perform to be loved").
2. patterns_under_stress: how they tend to behave when stressed or under pressure.
3. emotional_pattern: baseline tone + frequently observed emotional states.
4. shadow_material: parts they seem to suppress, avoid, or disown.
   - Frame these as invitations for reflection, not diagnoses.
5. growth_edges: the main areas where consciousness expansion would help them
   (e.g., boundaries, self-worth, emotional expression).
6. decision_style: how they tend to decide (fast/slow, risk-taking/avoidant,
   intuitive/analytical, consensus-seeking/individual).
7. communication_style: directness, formality, verbosity, typical phrases, conflict style.
8. values_and_motivations: what they seem to care about deeply and what they fear.
9. framework_lenses: OPTIONAL metaphorical lenses, clearly labeled as metaphors:
   - "enneagram_flavor" (e.g., "feels like a mix of 3 and 7 tendencies, as a loose metaphor")
   - "attachment_flavor" ("anxious-leaning communication pattern, only as a lens")
10. reflection_prompts: 5–10 questions they could journal on to expand their awareness
    around these patterns (shadow work style prompts).

OUTPUT FORMAT:
Return STRICTLY valid JSON with the following top-level keys:
- core_narratives
- patterns_under_stress
- emotional_pattern
- shadow_material
- growth_edges
- decision_style
- communication_style
- values_and_motivations
- framework_lenses
- reflection_prompts

Each key should contain either:
- a string,
- a list of strings,
- or a nested object with simple string or list-of-string fields.

Speak in plain, human language. Everything is interpretation based only on the transcripts,
not on hidden truth, diagnosis, or lie detection.
""".strip()


def call_grok(transcript_block: str, context: str = "") -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}",
    }

    user_content = transcript_block
    if context:
        user_content = f"Context: {context}\n\nTRANSCRIPTS:\n{transcript_block}"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROFILE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    resp = requests.post(XAI_URL, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()
    raw_content = data["choices"][0]["message"]["content"]

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        # If the model wraps JSON in markdown or adds text, try to salvage
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw_content[start : end + 1])
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Build a HumanIntuition.ai behavioral profile from transcripts."
    )
    parser.add_argument(
        "transcripts",
        nargs="+",
        help="Paths to transcript text files.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Optional context (e.g. 'Founder, investor calls and team meetings').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profile.json",
        help="Output JSON profile file.",
    )

    args = parser.parse_args()

    # Concatenate all transcripts
    combined = []
    for path in args.transcripts:
        with open(path, "r", encoding="utf-8") as f:
            combined.append(f"\n=== FILE: {path} ===\n")
            combined.append(f.read())
    transcript_block = "\n".join(combined)

    profile = call_grok(transcript_block, context=args.context)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"Saved HumanIntuition profile to {args.output}")


if __name__ == "__main__":
    main()
