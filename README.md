# HumanIntuition.ai

**What if you could read blindfolded?**

HumanIntuition.ai is an embodied-intelligence operating system that merges human consciousness, somatic awareness, intuition, and emotional attunement with machine-scale reasoning, prediction, and execution.

## Vision

Our path to AGI is to understand human consciousness and the bridge between human intuition and computer intuition.

## Mission

HumanIntuition.ai increases conscious awareness, somatic intelligence, relational attunement, emotional sovereignty, identity integration, and decision-making coherence. AI becomes an amplifier of clarity, not a replacement for human agency.

## Philosophy

Human intuition derives from somatic truth signals, emotional pattern tracking, energetic attunement, and unconscious pattern recognition. HumanIntuition.ai operates through a four-layer intelligence stack:

1. **Layer 1: Somatic & Emotional Intelligence** - Detect emotional activation, projection, avoidance, fragmentation, and energetic shifts
2. **Layer 2: Intuition & Relational Intelligence** - Pattern awareness, relational dynamics, developmental patterns, shadow tracking
3. **Layer 3: Cognitive Reasoning** - Analytical reasoning, pattern recognition, strategic coherence
4. **Layer 4: Strategic Execution** - Actionable pathways, embodied leadership, conscious alternatives

Most AI starts at Layer 3. HumanIntuition.ai begins at Layer 1.

The system analyzes how you actually show up in real conversations (calls, meetings, voice notes), builds a deep consciousness profile, then gives you a "superagent" version of yourself that helps you act from your most conscious, integrated self instead of autopilot patterns.

## Setup

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Create a `.env` file with your X.AI API key:
```
XAI_API_KEY=your_api_key_here
```

## Quick Start: Web App

For a simple web interface to upload and analyze transcripts or audio files:

```bash
# Install Flask (if not already installed)
pip3 install flask

# Optional: For audio transcription, install one of:
# pip3 install openai  # For OpenAI Whisper API
# pip3 install openai-whisper  # For local Whisper (no API key needed)

# Run the web app
python3 app.py
```

Then open your browser to `http://127.0.0.1:5001` and upload:
- **Text files**: `.txt` transcript files (analyzed directly)
- **Audio files**: `.m4a`, `.mp3`, `.wav`, `.mp4`, `.webm`, `.ogg`, `.flac` (automatically transcribed then analyzed)

**Audio Transcription Options:**
- **OpenAI Whisper API**: Requires `OPENAI_API_KEY` in `.env` (faster, uses API)
- **Local Whisper**: Requires `openai-whisper` package (no API key, but slower and needs disk space)

## Components

### 1. Profile Builder – `build_profile.py`

Builds a deep, non-clinical consciousness profile from one or more transcripts, integrating somatic awareness, emotional intelligence, relational dynamics, and shadow patterns.

```bash
python3 build_profile.py transcript1.txt transcript2.txt \
  --context "Founder on investor + team calls"
```

Outputs `profile.json` with:

* core narratives and beliefs
* patterns under stress
* emotional and communication patterns
* somatic awareness and energetic shifts
* shadow-material-style reflections (as interpretations, not facts)
* growth edges and more conscious alternatives
* metaphorical lenses (Enneagram/attachment *flavors*, not diagnoses)
* reflection prompts designed to expand consciousness

### 2. Emotional Mapping – `emotional_mapping.py`

Maps somatic and emotional patterns, shifts, and energetic coherence over a single transcript using Layer 1 (Somatic & Emotional Intelligence).

```bash
python3 emotional_mapping.py transcript1.txt
```

Outputs `emotional_map.json` with:

* timeline of segments, emotions, intensity, notes
* global summary, triggers, regulation style
* reflection prompts

### 3. Superagent – `superagent.py`

An embodied-intelligence agent that acts like "you, but more integrated" - operating from expanded consciousness, emotional sovereignty, and embodied leadership rather than autopilot patterns.

```bash
python3 superagent.py
```

You can ask:

* "How would my expanded self respond to this message?"
* "How should I talk to my team about this mistake?"
* "What would be a more conscious way to set this boundary?"

The agent:

* Matches your tone and values (based on `profile.json`)
* Operates from the four-layer intelligence stack (somatic → intuition → cognitive → strategic)
* Upgrades fear-based patterns (people-pleasing, conflict avoidance, projection, etc.) to more conscious alternatives
* Protects your sovereignty—amplifies intuition, reveals blind spots, but never replaces your agency
* Never diagnoses or claims to detect lies or trauma
* Frames everything as interpretation and support, not clinical truth

## Audio Sources

You can use transcripts from:
- **Zoom recordings** - Export audio, transcribe with Whisper
- **Google Meet** - Export audio, transcribe with Whisper
- **Otter.ai** - Already provides transcripts
- **Fireflies** - Already provides transcripts
- **Phantom** - Already provides transcripts
- **Apple Voice Memos** - Export audio, transcribe with Whisper

### Transcription Workflow

1. **If you have transcripts already** (Otter, Fireflies, etc.):
   - Export/save as `.txt` files
   - Upload directly in the web app or use with `build_profile.py`

2. **If you have audio files**:
   - **Option A (Recommended)**: Upload directly in the web app (`app.py`) - it will transcribe and analyze automatically
   - **Option B**: Use `transcribe_audio.py` to transcribe first, then use the transcript files
   - **Option C**: Use Whisper or another transcription service, save as `.txt`, then use `build_profile.py`

## Example Workflow

```bash
# Step 1: Get transcripts (from Otter, Fireflies, or transcribe audio)
# Save as transcript1.txt, transcript2.txt, etc.

# Step 2: Build profile from multiple conversations
python3 build_profile.py transcript1.txt transcript2.txt transcript3.txt \
  --context "Sales manager, 1:1s with team members"

# Step 3: Map emotions (optional)
python3 emotional_mapping.py transcript1.txt

# Step 4: Chat with your expanded consciousness agent
python3 superagent.py
```

## Safety & Ethics

* This system does **not** perform mental health diagnosis.
* It does **not** claim to detect trauma, attachment style, or truthfulness as fact.
* All outputs are interpretive and should be used as tools for consciousness expansion and reflection, not as final judgments about you or anyone else.
* The system protects user sovereignty—it amplifies clarity and reveals blind spots, but never replaces human agency.
* Get explicit consent if analyzing other people's audio.
* Be transparent that this is interpretive, not factual.
* Use as a consciousness-expansion and self-awareness tool, not a diagnostic tool.

## Files

- `app.py` - **Web app for bulk transcript upload** (Flask)
- `main.py` - Single conversation analysis (terminal script)
- `build_profile.py` - Deep behavioral/consciousness profile builder
- `emotional_mapping.py` - Map emotions to transcripts
- `superagent.py` - Expanded consciousness agent (HumanIntuition agent)
- `transcribe_audio.py` - Audio transcription helper (Whisper integration)
- `requirements.txt` - Python dependencies
- `.env` - API keys (not in git)
