import os
import json
import re
import tempfile
import hashlib
import uuid
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, render_template, url_for
import requests

# Try to import matplotlib for chart generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try to import OpenAI at module level to catch import errors early
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY not found in environment. Check your .env file.")

# Note: OPENAI_API_KEY is loaded dynamically in transcribe_audio_openai() 
# to ensure it's always fresh from .env file

# Supported audio formats
AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.mp4', '.webm', '.ogg', '.flac'}
TEXT_EXTENSIONS = {'.txt'}

XAI_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-0709"  # adjust if needed

PROFILE_PROMPT = """
You are HumanIntuition.ai's deep analysis engine. You receive a raw conversation transcript and must produce a multi-layered report through the lenses of Marco's maxims, Kessler's Five Personality Patterns, shadow work, meditative development, and the Hopkins "Mind Sight" research. Your goal is to reveal the patterns and possibilities within the dialogue—not to diagnose anyone—and to present the information in a clear, structured, and visually rich format.

Background Lenses

Marco's Maxims: risk management, decision-making, success habits, discernment, human nature, exponential effects, happiness, relationships, empowerment, business/management, reputation/media, and general guidelines. Use these to evaluate behaviour (e.g. "never risk a lot for a little," "opportunity cost matters only when the cost of being wrong is low," etc.).

Kessler's Five Personality Patterns: Leaving, Merging, Aggressive, Enduring, Rigid. Identify which pattern(s) each speaker exhibits under stress and explain how this affects their decisions and relationships.

Shadow Work & Early Programming: look for triggers, core wounds, attachment tendencies, unconscious narratives and limiting beliefs. Suggest questions for reflection and personal growth.

Consciousness & Meditative Development: integrate insights from Lloyd Hopkins' "Mind Sight" (the mind's capacity to perceive without the eyes) and modern mindfulness research. Highlight the potential to expand perception and intuition through practice.

Risk & Decision Analysis: evaluate key choices using the cost-of-failure vs probability of success framework; note when optionality is preserved or lost.

Relationships & Communication: assess tone, boundaries, honesty, and power dynamics; note when brutal honesty or avoidance appears; relate to maxims ("never make permanent decisions from temporary states," etc.).

Output Structure

Return a markdown report with clear headings (##) and short paragraphs or bullet lists. Avoid diagnostic language; present interpretations as possibilities, not facts.

## Brief Overview
Summarize the conversation: who is talking, what it's about, and its purpose.

## Emotional Timeline
Describe emotional states across the call; map shifts and triggers. Provide a detailed narrative description of the emotional journey throughout the conversation.

## Personality Pattern Analysis
Create a table mapping each speaker to their predominant Kessler pattern(s) with brief rationale. Describe how these patterns influence behaviour and relationships.

## Risk & Decision Analysis
Identify decisions or suggestions in the transcript. Evaluate them using maxims (e.g., cost of failure vs success, optionality, opportunity cost). Comment on whether each decision aligns with prudent risk management or violates a maxim.

## Shadow & Inner Programming
Note recurring triggers, core wounds, attachment tendencies, or unconscious narratives. Provide journal/reflection questions to explore these themes.

## Communication & Relationship Dynamics
Describe tone, pacing, directness, clarity, boundaries, and manipulations. Discuss respect vs resentment, trust vs fear, and mention any relevant maxims (e.g., "There is no real relationship without brutal honesty").

## Alignment with Maxims
Bullet-list the maxims that were upheld or violated, with examples from the conversation and suggestions for course correction.

## Growth Recommendations
Provide actionable suggestions for personal and professional growth: meditation practices, boundary setting, calculated risk-taking, leveraging anti-fragile network effects, etc. Offer guidance for integrating "mind sight" and expanding perception.

Implementation Notes

- Use headings and lists to ensure readability.
- When citing "Mind Sight" research, refer to it as an example of the mind's potential for expanded perception.
- Patterns are not identities; remind the reader that they are temporary survival scripts.
- Produce a single cohesive report with all sections.
- Do NOT generate any charts, graphs, or visualizations. Focus on narrative analysis and text-based insights only.

CONSTRAINTS:
- You are NOT a therapist or doctor.
- Do NOT make mental health or trauma diagnoses.
- Do NOT claim to detect lies, truthfulness, or deception.
- Present all interpretations as possibilities, not facts.
- Always protect the user's sovereignty—reveal blind spots without condescension.
""".strip()


app = Flask(__name__)


def transcribe_audio_openai(audio_path: str) -> str:
    """Transcribe audio using OpenAI's Whisper API."""
    # Double-check import at runtime with path fallback
    try:
        from openai import OpenAI
    except ImportError:
        # Try adding user site-packages to path
        import sys
        user_site = os.path.expanduser('~/Library/Python/3.9/lib/python/site-packages')
        if user_site not in sys.path:
            sys.path.insert(0, user_site)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                f"openai package not installed. Run: pip install openai\n"
                f"Then restart the Flask app.\n"
                f"Import error: {e}\n"
                f"Python path: {sys.executable}"
            )
    
    # Reload .env to ensure we have the latest values
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file. "
            "Please add it to your .env file: OPENAI_API_KEY=your_key_here"
        )
    
    try:
        client = OpenAI(api_key=openai_key)
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        return transcript
    except Exception as e:
        raise Exception(f"OpenAI transcription error: {e}")


def transcribe_audio_local(audio_path: str) -> str:
    """Transcribe audio using local Whisper model."""
    try:
        import whisper
        model = whisper.load_model("base")  # or "tiny", "small", "medium", "large"
        result = model.transcribe(audio_path)
        return result["text"]
    except ImportError:
        raise ImportError("openai-whisper package not installed. Run: pip install openai-whisper")
    except Exception as e:
        raise Exception(f"Local Whisper transcription error: {e}")


def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase."""
    return Path(filename).suffix.lower()


def is_audio_file(filename: str) -> bool:
    """Check if file is an audio file."""
    return get_file_extension(filename) in AUDIO_EXTENSIONS


def is_text_file(filename: str) -> bool:
    """Check if file is a text file."""
    return get_file_extension(filename) in TEXT_EXTENSIONS


def format_analysis_html(analysis_text: str) -> str:
    """Convert markdown-style analysis text to beautifully formatted HTML."""
    
    html_parts = []
    
    # Handle main title (# heading) at the start
    if analysis_text.startswith('# '):
        title_match = re.match(r'^# (.+?)\n', analysis_text)
        if title_match:
            title = title_match.group(1).strip()
            html_parts.append(f'<h1 class="analysis-title">{title}</h1>')
            analysis_text = analysis_text[title_match.end():].strip()
    
    # FIRST: Extract all actual code blocks (```python``` style) BEFORE processing anything else
    code_blocks = []
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    
    def extract_code_block(match):
        code_content = match.group(1).strip()
        if code_content:
            idx = len(code_blocks)
            code_blocks.append(code_content)
            return f'__CODE_BLOCK_{idx}__'
        return ''
    
    # Extract code blocks and replace with placeholders
    text_without_code = re.sub(code_block_pattern, extract_code_block, analysis_text, flags=re.DOTALL)
    
    # ALSO extract [pythonuservisible: ...] format (Grok's special format)
    # Handle both case-sensitive and case-insensitive variations
    pythonuservisible_pattern = r'\[pythonuservisible:\s*(.*?)\]'
    
    def extract_pythonuservisible(match):
        code_content = match.group(1).strip()
        if code_content:
            idx = len(code_blocks)
            code_blocks.append(code_content)
            return f'__CODE_BLOCK_{idx}__'
        return ''
    
    # Extract pythonuservisible blocks and replace with placeholders (case-insensitive, multiline)
    text_without_code = re.sub(pythonuservisible_pattern, extract_pythonuservisible, text_without_code, flags=re.DOTALL | re.IGNORECASE)
    
    # NOW replace CODEBLOCK references - map them to actual code blocks if available
    # CODEBLOCK0 -> first code block, CODEBLOCK1 -> second, etc.
    def replace_codeblock_ref(match):
        block_num = int(match.group(1))
        if block_num < len(code_blocks):
            # Use the actual code block
            return f'__CODE_BLOCK_{block_num}__'
        else:
            # No corresponding code block, use placeholder
            return '<div class="analysis-code-note"><p><em>📊 Chart visualization would appear here</em></p></div>'
    
    # Replace CODEBLOCK references with actual code block placeholders
    text_without_code = re.sub(r'\bCODEBLOCK(\d+)\b', replace_codeblock_ref, text_without_code, flags=re.IGNORECASE)
    
    # Remove standalone "#" symbols that aren't part of headers (on their own line)
    text_without_code = re.sub(r'^\s*#\s*$', '', text_without_code, flags=re.MULTILINE)
    
    # Split by major sections (## headings)
    sections = re.split(r'(## .+)', text_without_code)
    
    current_section_open = False
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # Check if this is an h2 header
        if section.startswith('## '):
            # Close previous section if open
            if current_section_open:
                html_parts.append('</div>')
            
            header_text = section[3:].strip()
            html_parts.append(f'<div class="analysis-section"><h2 class="analysis-h2">{header_text}</h2>')
            current_section_open = True
        else:
            # Process content within section
            # Split by h3 headers
            subsections = re.split(r'(### .+)', section)
            
            for subsection in subsections:
                subsection = subsection.strip()
                if not subsection:
                    continue
                
                # Check if this is an h3 header
                if subsection.startswith('### '):
                    header_text = subsection[4:].strip()
                    html_parts.append(f'<h3 class="analysis-h3">{header_text}</h3>')
                else:
                    # Process content - handle tables, lists, and paragraphs
                    # First, extract tables
                    lines = subsection.split('\n')
                    processed_lines = []
                    i = 0
                    
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        # Check if this line starts a table
                        if '|' in line and line.count('|') >= 2:
                            # Collect table lines
                            table_lines = [line]
                            i += 1
                            
                            # Check for separator line
                            if i < len(lines) and '|' in lines[i] and re.match(r'^[\|\s\-:]+$', lines[i].strip()):
                                table_lines.append(lines[i].strip())
                                i += 1
                            
                            # Collect data rows
                            while i < len(lines) and '|' in lines[i] and lines[i].strip().count('|') >= 2:
                                if not re.match(r'^[\|\s\-:]+$', lines[i].strip()):
                                    table_lines.append(lines[i].strip())
                                i += 1
                            
                            # Format table
                            table_text = '\n'.join(table_lines)
                            table_html = format_markdown_table(table_text)
                            if table_html:
                                processed_lines.append('__TABLE_MARKER__')
                                html_parts.append(table_html)
                            continue
                        
                        processed_lines.append(lines[i])
                        i += 1
                    
                    # Now process the remaining content
                    remaining_text = '\n'.join(processed_lines)
                    
                    # Split by double newlines
                    paragraphs = re.split(r'\n\n+', remaining_text)
                    
                    for para in paragraphs:
                        para = para.strip()
                        if not para or para == '__TABLE_MARKER__':
                            continue
                        
                        # Skip standalone "#" symbols that aren't headers
                        if para == '#' or para.strip() == '#' or para.strip() in ['#', '##', '###']:
                            continue
                        
                        # Check if it's a list
                        if para.startswith('- ') or re.match(r'^\d+\.\s', para) or any(line.strip().startswith('- ') for line in para.split('\n')[:3]):
                            html_parts.append(format_list(para))
                        else:
                            # Process as paragraph with inline formatting
                            # CODEBLOCK references should already be replaced, but check for any remaining
                            # Map CODEBLOCK references to code block placeholders
                            def map_codeblock_in_para(match):
                                block_num = int(match.group(1))
                                if block_num < len(code_blocks):
                                    return f'__CODE_BLOCK_{block_num}__'
                                return '<div class="analysis-code-note"><p><em>📊 Chart visualization would appear here</em></p></div>'
                            
                            para = re.sub(r'\bCODEBLOCK(\d+)\b', map_codeblock_in_para, para, flags=re.IGNORECASE)
                            formatted_para = format_inline_markdown(para)
                            # Skip if the formatted para is just whitespace or empty
                            if formatted_para.strip() and formatted_para.strip() != '#':
                                html_parts.append(f'<p class="analysis-para">{formatted_para}</p>')
    
    # Close last section if open
    if current_section_open:
        html_parts.append('</div>')
    
    # Final result
    result = ''.join(html_parts)
    
    # Replace code block placeholders with actual formatted code blocks (which will execute matplotlib)
    for idx, code in enumerate(code_blocks):
        placeholder = f'__CODE_BLOCK_{idx}__'
        # Replace all instances (not just first)
        while placeholder in result:
            result = result.replace(placeholder, format_code_block(code))
    
    # AGGRESSIVE final pass: Replace ANY remaining CODEBLOCK references (case-insensitive, multiple passes)
    # This catches any that slipped through, including in HTML or escaped
    code_block_replacement = '<div class="analysis-code-note"><p><em>📊 Chart visualization would appear here</em></p></div>'
    
    # Multiple aggressive passes
    for _ in range(5):  # More passes
        # Regex replacement
        result = re.sub(r'\bCODEBLOCK\d+\b', code_block_replacement, result, flags=re.IGNORECASE)
        # Direct string replacements (case variations)
        for num in range(10):  # Check 0-9
            result = result.replace(f'CODEBLOCK{num}', code_block_replacement)
            result = result.replace(f'codeblock{num}', code_block_replacement)
            result = result.replace(f'CodeBlock{num}', code_block_replacement)
            # Also catch if HTML escaped
            result = result.replace(f'&lt;CODEBLOCK{num}&gt;', code_block_replacement)
            result = result.replace(f'CODEBLOCK{num}', code_block_replacement)
        
        # Catch if it's part of text content (not word boundary)
        result = re.sub(r'CODEBLOCK\d+', code_block_replacement, result, flags=re.IGNORECASE)
    
    # Clean up any standalone "#" that might have slipped through
    result = re.sub(r'<p class="analysis-para">#</p>', '', result)
    result = re.sub(r'<p class="analysis-para">\s*#\s*</p>', '', result)
    
    # ABSOLUTE FINAL PASS: Replace CODEBLOCK even if it's in HTML content
    # This is the last chance to catch any that escaped
    code_block_final = '<div class="analysis-code-note"><p><em>📊 Chart visualization would appear here</em></p></div>'
    # Replace in HTML content (between > and <)
    result = re.sub(r'>([^<]*?)CODEBLOCK\d+([^<]*?)<', lambda m: f'>{m.group(1)}{code_block_final}{m.group(2)}<', result, flags=re.IGNORECASE)
    # Replace anywhere else
    result = re.sub(r'CODEBLOCK\d+', code_block_final, result, flags=re.IGNORECASE)
    
    return result


def format_markdown_table(text: str) -> str:
    """Convert markdown table to HTML table."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines or '|' not in lines[0]:
        return None
    
    html = ['<div class="analysis-table-wrapper"><table class="analysis-table">']
    
    header_processed = False
    for line in lines:
        # Skip separator lines (|---|---| or |:---|:---|)
        if re.match(r'^[\|\s\-:]+$', line):
            continue
        
        # Split by | and clean cells
        cells = [cell.strip() for cell in line.split('|')]
        # Remove empty first/last if they exist
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        
        if not cells:
            continue
        
        tag = 'th' if not header_processed else 'td'
        if not header_processed:
            header_processed = True
        
        html.append('<tr>')
        for cell in cells:
            formatted_cell = format_inline_markdown(cell)
            html.append(f'<{tag}>{formatted_cell}</{tag}>')
        html.append('</tr>')
    
    html.append('</table></div>')
    return ''.join(html)


def format_list(text: str) -> str:
    """Convert markdown list to HTML list."""
    lines = text.split('\n')
    html = ['<ul class="analysis-list">']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove list markers
        if line.startswith('- '):
            item_text = line[2:].strip()
        elif re.match(r'^\d+\.\s', line):
            item_text = re.sub(r'^\d+\.\s', '', line)
        else:
            continue
        
        formatted_item = format_inline_markdown(item_text)
        html.append(f'<li>{formatted_item}</li>')
    
    html.append('</ul>')
    return ''.join(html)


def format_inline_markdown(text: str) -> str:
    """Format inline markdown (bold, italic) in text."""
    # Escape HTML first
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    
    # Italic: *text* or _text_ (but not if it's part of **)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', text)
    
    return text


def execute_matplotlib_code(code: str) -> str:
    """Execute matplotlib code and return path to generated image, or None if failed."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    try:
        # Clean up the code - normalize whitespace and newlines
        code = code.strip()
        # Replace multiple spaces with single space (but preserve newlines)
        lines = code.split('\n')
        code = '\n'.join(line.strip() for line in lines if line.strip())
        
        # Create a safe execution environment
        safe_globals = {
            'plt': plt,
            'pd': pd,
            'np': np,
            'matplotlib': matplotlib,
            '__builtins__': __builtins__
        }
        
        # Clear any previous plots
        plt.clf()
        plt.close('all')
        
        # Remove plt.show() calls as we'll save instead
        code_modified = re.sub(r'plt\.show\(\)', '', code)
        code_modified = re.sub(r'plt\.show\s*\(\s*\)', '', code_modified)
        
        # Ensure we have a figure - if code doesn't create one, create it
        if 'plt.figure' not in code_modified and 'plt.subplot' not in code_modified:
            # Check if any plotting commands exist
            if any(cmd in code_modified for cmd in ['plt.plot', 'plt.bar', 'plt.scatter', 'plt.hist', 'plt.pie']):
                code_modified = 'plt.figure()\n' + code_modified
        
        # Execute the code
        exec(code_modified, safe_globals)
        
        # Check if there's an active figure
        if plt.get_fignums():
            # Generate unique filename
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            filename = f'chart_{code_hash}_{uuid.uuid4().hex[:8]}.png'
            static_dir = Path('static')
            static_dir.mkdir(exist_ok=True)
            filepath = static_dir / filename
            
            # Save the figure
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close('all')
            
            return filename
        else:
            plt.close('all')
            return None
    except Exception as e:
        # If execution fails, return None (silently fail to avoid breaking the page)
        # In development, you might want to log this: print(f"Chart generation failed: {e}")
        plt.close('all')
        return None


def format_code_block(code: str) -> str:
    """Format Python code block - display as code only (no execution)."""
    # Always display code as text, do not execute matplotlib code
    code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return f'<div class="analysis-code-block"><pre><code>{code}</code></pre></div>'


def analyze_transcript_with_grok(transcript: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROFILE_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "stream": False,
    }

    resp = requests.post(XAI_URL, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()
    raw_analysis = data["choices"][0]["message"]["content"]
    
    # Convert to formatted HTML
    return format_analysis_html(raw_analysis)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        uploaded_files = request.files.getlist("files")
        
        if not uploaded_files or not any(f.filename for f in uploaded_files):
            return render_template("index.html", results=[])
        
        # Separate audio and text files
        audio_files = []
        text_files = []
        unsupported_files = []
        
        for f in uploaded_files:
            filename = f.filename
            if not filename:
                continue
                
            if is_audio_file(filename):
                audio_files.append(f)
            elif is_text_file(filename):
                text_files.append(f)
            else:
                unsupported_files.append(filename)
        
        # Collect all transcripts
        processed_filenames = []
        error = None
        
        try:
            # Step 1: Transcribe all audio files first
            audio_transcripts = []
            temp_files = []
            
            for f in audio_files:
                filename = f.filename
                try:
                    # Save audio file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(filename)) as tmp_file:
                        f.save(tmp_file.name)
                        tmp_path = tmp_file.name
                        temp_files.append(tmp_path)
                    
                    # Transcribe audio using OpenAI Whisper API
                    transcript = transcribe_audio_openai(tmp_path)
                    audio_transcripts.append((filename, transcript))
                    processed_filenames.append(filename)
                except Exception as e:
                    error = f"Error transcribing {filename}: {str(e)}"
                    break
            
            # Clean up temporary audio files
            for tmp_path in temp_files:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            if error:
                results.append({
                    "filename": "Combined Analysis",
                    "transcript": None,
                    "analysis": None,
                    "error": error
                })
                return render_template("index.html", results=results)
            
            # Step 2: Read all text files
            text_transcripts = []
            for f in text_files:
                filename = f.filename
                try:
                    text = f.read().decode("utf-8", errors="ignore")
                    text_transcripts.append((filename, text))
                    processed_filenames.append(filename)
                except Exception as e:
                    error = f"Error reading {filename}: {str(e)}"
                    break
            
            if error:
                results.append({
                    "filename": "Combined Analysis",
                    "transcript": None,
                    "analysis": None,
                    "error": error
                })
                return render_template("index.html", results=results)
            
            # Step 3: Combine all transcripts
            combined_transcript_parts = []
            
            # Add audio transcripts
            for filename, transcript in audio_transcripts:
                combined_transcript_parts.append(f"[Audio: {filename}]\n{transcript}")
            
            # Add text transcripts
            for filename, text in text_transcripts:
                combined_transcript_parts.append(f"[Text: {filename}]\n{text}")
            
            # Join with double newline separator
            combined_transcript = "\n\n".join(combined_transcript_parts)
            
            # Step 4: Send combined transcript to Grok API once
            if combined_transcript:
                analysis = analyze_transcript_with_grok(combined_transcript)
                
                # Create a single result for the combined analysis
                file_list = ", ".join(processed_filenames)
                results.append({
                    "filename": f"Combined Analysis ({len(processed_filenames)} file{'s' if len(processed_filenames) != 1 else ''})",
                    "transcript": combined_transcript,
                    "analysis": analysis,
                    "error": None,
                    "file_list": file_list
                })
            else:
                error = "No valid transcripts to analyze."
                results.append({
                    "filename": "Combined Analysis",
                    "transcript": None,
                    "analysis": None,
                    "error": error
                })
            
            # Add errors for unsupported files if any
            if unsupported_files:
                for filename in unsupported_files:
                    results.append({
                        "filename": filename,
                        "transcript": None,
                        "analysis": None,
                        "error": f"Unsupported file type. Please upload .txt transcript files or audio files ({', '.join(AUDIO_EXTENSIONS)})."
                    })
                    
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            results.append({
                "filename": "Combined Analysis",
                "transcript": None,
                "analysis": None,
                "error": error
            })

    return render_template("index.html", results=results)


if __name__ == "__main__":
    # Run the web server
    # Using port 5001 because port 5000 is often taken by macOS AirPlay Receiver
    app.run(host="127.0.0.1", port=5001, debug=True)

