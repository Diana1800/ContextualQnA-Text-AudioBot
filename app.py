import os
import whisper
from flask import Flask, request, render_template_string, session, jsonify
from werkzeug.utils import secure_filename
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(temperature=0, model='gpt-4o')

# Load the Whisper model
model = whisper.load_model("base")

# Load the content of from TXT file
with open('/home/diana/Documents/me.txt', 'r', encoding='utf-8') as file:
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

# Allowed extensions for audio files
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# Helper function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/', methods=['GET', 'POST'])
def chat():

    if request.method == 'POST':
        user_input = request.form.get('user_input', None)
        audio_file = request.files.get('audio_file', None)
        print(f"Received user input: {user_input}")
        print(f"Received audio file: {audio_file.filename if audio_file else 'No audio file uploaded'}")

        transcribed_audio = None

        # Check if there is an audio file
        if audio_file and allowed_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            audio_path = os.path.join('/tmp', filename)
            audio_file.save(audio_path)                       # Save the file temporarily
            print(f"Saved audio file to: {audio_path}")

            # Load and transcribe audio using Whisper locally
            try:
                audio = whisper.load_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)

                # Detect language 
                _, probs = model.detect_language(mel)
                print(f"Detected language: {max(probs, key=probs.get)}")

                # Decode the audio
                options = whisper.DecodingOptions()
                result = whisper.decode(model, mel, options)
                transcribed_audio = result.text
                print(f"Transcribed audio: {transcribed_audio}")
            except Exception as e:
                print(f"Error processing audio with Whisper: {e}")

        # Use the transcribed audio as input if available
        if transcribed_audio:
            user_input = transcribed_audio

        if not user_input:
            print("Error: No input provided")                  # Debugging print
            return jsonify({'response': 'Error: No input provided'}), 400

        if 'chat_history' not in session:
            session['chat_history'] = []

        # Combine the data into context
        context = all_data    
        print(f"Context Length: {len(context)}")

        # Generate an answer using LangChain (Replace this with your own logic)
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        runnable_sequence = prompt | llm  
        answer = runnable_sequence.invoke({"context": context, "question": user_input})

        print(f"Generated Answer: {answer.content}")

        #session['chat_history'].append(f"You: {user_input}<br>AI: {answer.content}<br>")
        session['chat_history'].append(
            f'<span style="color: orange;">You:</span> {user_input}<br>'
            f'<span style="color: green;">AI:</span> {answer.content}<br>')

        chat_history = "<br>".join(session['chat_history'])
        session.modified = True 
        return jsonify({'response': chat_history})

    # Initialize chat history for GET requests
    # session['chat_history'] = []
    return render_template_string(template)

if __name__ == '__main__':
    app.run(port=5000)
