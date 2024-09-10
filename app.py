import os
import whisper
from flask import Flask, request, render_template_string, session, jsonify
from werkzeug.utils import secure_filename
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize OpenAI model (gpt-4o for LangChain)
llm = ChatOpenAI(temperature=0, model='gpt-4o')

# Load the Whisper model for audio transcription
model = whisper.load_model("base")

# Load the content from a text file
with open('context.txt', 'r', encoding='utf-8') as file:
    all_data = file.read()

# Define the prompt template for LangChain
prompt_template = """
Given the following context:
{context}

Answer the following question:
{question}
"""

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Save session

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# Helper function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form.get('user_input', None)
        audio_file = request.files.get('audio_file', None)
        transcribed_audio = None

        # Handle audio file: Only process audio if uploaded
        if audio_file and allowed_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            audio_path = os.path.join('/tmp', filename)
            audio_file.save(audio_path)
            print(f"Saved audio file to: {audio_path}")

            # Convert audio to .wav if necessary
            try:
                if audio_path.endswith('.m4a'):
                    wav_path = audio_path.replace('.m4a', '.wav')
                    AudioSegment.from_file(audio_path).export(wav_path, format='wav')
                    audio_path = wav_path
                    print(f"Converted m4a to wav: {audio_path}")

                # Transcribe audio using Whisper
                audio = whisper.load_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                result = whisper.decode(model, mel, whisper.DecodingOptions())
                transcribed_audio = result.text
                print(f"Transcribed audio: {transcribed_audio}")
            except Exception as e:
                print(f"Error processing audio: {e}")
                return jsonify({'response': f"Error processing audio: {str(e)}"}), 500

        # If transcribed audio exists and there's no text input, use the transcribed audio
        if transcribed_audio and not user_input:
            user_input = transcribed_audio
            print(f"Using transcribed audio as user input: {user_input}")

        # If neither input nor audio is available, return an error
        if not user_input:
            print("Error: No input provided")
            return jsonify({'response': 'Error: No input provided'}), 400

        # Initialize session if not already done
        if 'chat_history' not in session:
            session['chat_history'] = []

        # Generate an answer using LangChain
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        runnable_sequence = prompt | llm
        answer = runnable_sequence.invoke({"context": all_data, "question": user_input})
        print(f"Generated Answer: {answer.content}")

        # Append chat history
        session['chat_history'].append(
            f'<span style="color: orange;">You:</span> {user_input}<br>'
            f'<span style="color: green;">AI:</span> {answer.content}<br>'
        )
        print("Appended to chat history.")

        # Ensure session is updated
        chat_history = "<br>".join(session['chat_history'])
        session.modified = True  # Force session save
        print("Session modified and chat history prepared.")

        return jsonify({'response': chat_history})

    return render_template_string(template)

if __name__ == '__main__':
    app.run(port=5000)
