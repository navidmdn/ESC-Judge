import os
from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# Directory containing transcripts
TRANSCRIPTS_DIR = "output/test"

def load_transcripts():
    """Load available transcript filenames from the directory."""
    return [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".txt")]

def read_transcript(data_dir, filename):
    """Read the content of a transcript file and format it as separate chat boxes."""
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):

        return f"Transcript not found in {file_path}"

    chat_html = []
    with open(file_path, "r", encoding="utf-8") as file:
        conversation_text = file.read().strip()
        pattern = r"(seeker:|supporter:)(.*?)(?=(seeker:|supporter:|$))"
        matches = re.findall(pattern, conversation_text, re.DOTALL)
        conversation = [(speaker.strip(), message.strip()) for speaker, message, _ in matches]

        for speaker, message in conversation:
            if speaker.startswith("seeker"):
                chat_html.append(f'<div class="chat seeker"><strong>🧑 Seeker:</strong> {message.strip()}</div>')
            elif speaker.startswith("supporter"):
                chat_html.append(f'<div class="chat helper"><strong>🤖 Helper:</strong> {message.strip()}</div>')

    return "\n".join(chat_html)
@app.route("/")
def index():
    """Render the main page with dropdown selection."""
    transcripts = load_transcripts()
    return render_template("offline_vis.html", transcripts=transcripts)

@app.route("/get_transcript", methods=["POST"])
def get_transcript():
    """Fetch and return a transcript in raw Markdown format."""
    data = request.json
    filename = data.get("filename")
    transcript_content = read_transcript(TRANSCRIPTS_DIR, filename)
    return jsonify({"content": transcript_content})

if __name__ == "__main__":
    app.run(debug=True)
