from flask import Flask, render_template
from flask_socketio import SocketIO
from fire import Fire
import os
from evaluation.langchain_persona_based_chat import run_simulation

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load OpenAI API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')

user_args = {}

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('start_simulation')
def handle_start_simulation():
    """Handles the start simulation event triggered by the client."""
    if not user_args:
        socketio.emit('simulation_error', {'message': 'Simulation parameters not set!'})
        return

    print("Starting simulation with user-provided arguments...")

    socketio.start_background_task(
        target=run_simulation,
        supporter_persona_file=user_args['supporter_persona_file'],
        seeker_personas_file=user_args['seeker_personas_file'],
        supporter_llm_name=user_args['supporter_llm_name'],
        seeker_llm_name=user_args['seeker_llm_name'],
        temperature=user_args['temperature'],
        max_new_tokens=user_args['max_new_tokens'],
        hf=user_args['hf'],
        load_in_4bit=user_args['load_in_4bit'],
        cache_dir=user_args['cache_dir'],
        output_prefix=user_args['output_prefix'],
        visualization_socket=socketio,
    )

    socketio.emit('simulation_started', {'message': 'Simulation has started!'})

def run(
        supporter_persona_file: str = 'p1-test.txt', seeker_personas_file: str = 'data/test_persona.json',
        supporter_llm_name: str = 'gpt-4o', seeker_llm_name: str = 'gpt-4o', temperature=0.8, max_new_tokens=4096,
        hf=False, load_in_4bit=False,
        cache_dir=None, output_prefix="conv"):

    # todo: add personality dropdown + supporter/helper model dropdowns
    global user_args
    user_args = {
        'supporter_persona_file': supporter_persona_file,
        'seeker_personas_file': seeker_personas_file,
        'supporter_llm_name': supporter_llm_name,
        'seeker_llm_name': seeker_llm_name,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
        'hf': hf,
        'load_in_4bit': load_in_4bit,
        'cache_dir': cache_dir,
        'output_prefix': output_prefix
    }

    print("Simulation parameters set. Waiting for user to start simulation via the UI...")

    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    Fire(run)
