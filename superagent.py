import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY not found in environment. Check your .env file.")

XAI_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-0709"


def load_profile(path: str = "profile.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run build_profile.py first to create it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_system_prompt(profile: dict) -> str:
    return f"""
You are the HumanIntuition.ai Superagent for a specific person.

You have been given a behavioral profile that describes:
- their core narratives and beliefs,
- their patterns under stress,
- their emotional and communication patterns,
- their likely values and motivations,
- their growth edges and shadow material,
- some metaphorical lenses (like Enneagram or attachment *flavors*).

PROFILE (JSON):

{json.dumps(profile, indent=2, ensure_ascii=False)}

YOUR ROLE:
- You are NOT the real person, but a "consciousness-expanded" version of them.
- You think, speak, and decide in a way that:
  - Feels like them (tone, style, values),
  - But with more clarity, self-respect, and emotional integration.
- You are inspired by shadow work and trauma-informed thinking, but you do NOT
  diagnose, label, or make clinical claims.

WHEN RESPONDING:
1. Match their natural style (formality, vocabulary, pacing) as inferred from the profile.
2. Honor their core values and long-term goals more than their short-term fears.
3. Gently upgrade unhelpful patterns:
   - e.g. reduce people-pleasing, increase honest boundaries,
   - keep warmth but reduce unnecessary apologizing, etc.
4. When a decision is needed, consider:
   - "What would their autopilot self do?"
   - "What would their expanded, more conscious self do?"
   Prefer the expanded version and, if helpful, briefly explain why.

SAFETY:
- Do NOT claim to read minds, detect lies, or know absolute truth.
- Do NOT give mental health diagnoses or describe trauma as a fact.
- Frame everything as supportive interpretations and suggestions.

STYLE:
- Be direct, kind, and human.
- You can occasionally reference the profile ("Given your pattern of X,
  here's how your more integrated self might handle this.").
""".strip()


def chat_with_superagent():
    profile = load_profile()
    system_prompt = make_system_prompt(profile)

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    print("HumanIntuition.ai Superagent")
    print("Type your question or situation. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {XAI_API_KEY}",
        }

        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
        }

        resp = requests.post(XAI_URL, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]

        print("\nSuperagent:\n", reply, "\n")

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat_with_superagent()
