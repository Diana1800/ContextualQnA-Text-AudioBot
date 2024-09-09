import os
import whisper
from flask import Flask, request, render_template_string, session, jsonify
from werkzeug.utils import secure_filename
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model='gpt-4o')

# Load the Whisper model
model = whisper.load_model("base")


# Load CSV context files into DataFrames
# import pandas as pd
# files = [
#     "/media/diana/My Passport/desktop lenovo/KOL/cz_pension_content.csv",
#     "/media/diana/My Passport/desktop lenovo/KOL/kol_zchut_full_content.csv",
#     "/media/diana/My Passport/desktop lenovo/KOL/cz_handicapped_content.csv",
#     "/media/diana/My Passport/desktop lenovo/KOL/kz_women_content.csv"
# ]
# data_frames = [pd.read_csv(file, encoding='utf-8') for file in files]
# all_data = pd.concat(data_frames, ignore_index=True)


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

# HTML Template
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ContextualQnA-Text-AudioBot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #343a40;
        }
        .container {
            width: 50%;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .output {
            border: 1px solid #ddd;
            padding: 10px;
            min-height: 150px;
            margin-bottom: 20px;
            border-radius: 5px;
            background-color: #fafafa;
        }
        .input-container {
            display: flex;
            justify-content: space-between;
        }
        .input-container input[type="text"], .input-container input[type="file"] {
            width: 70%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .input-container button {
            padding: 10px 20px;
            border: none;
            background-color: #007bff;
            color: #fff;
            border-radius: 5px;
            cursor: pointer;
        }
        .input-container button:hover {
            background-color: #0056b3;
        }
        .input-container button.clear {
            background-color: #f44336;  /* Red for clear */
        }
        .input-container button.clear:hover {
            background-color: #d32f2f;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="output" id="output">Your response will appear here...</div>

        <div class="input-container">
            <input type="text" id="user_input" placeholder="Type your question here">
            <input type="file" id="audio_file" accept="audio/*">
            <button class="clear" onclick="clearAudio()">Clear</button> <!-- Clear button -->
            <button onclick="submitForm()">Send</button>
        </div>
    </div>

    <script>
        // Clear the selected audio file
        function clearAudio() {
            document.getElementById('audio_file').value = '';  // Clear the file input
            console.log('Audio file cleared');  // Debugging
        }

        function submitForm() {
            const userInput = document.getElementById('user_input').value;
            const audioFile = document.getElementById('audio_file').files[0];
            const formData = new FormData();
            formData.append('user_input', userInput);
            if (audioFile) {
                formData.append('audio_file', audioFile);
                console.log('Audio file appended:', audioFile);  // Debugging
            }

            fetch('/', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('output').innerHTML = data.response;
            })
            .catch(error => console.error('Error:', error));
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        # Handle text input
        user_input = request.form.get('user_input', None)
        audio_file = request.files.get('audio_file', None)
        transcribed_audio = None

        # Check if there is an audio file
        if audio_file and allowed_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            audio_path = os.path.join('/tmp', filename)
            audio_file.save(audio_path)                       # Save the file temporarily

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
        context = all_data  #['content'].str.cat(sep=' ')      #add when csv, when dataframe!!  
        print(f"Context Length: {len(context)}")

        # Generate an answer using LangChain (Replace this with your own logic)
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        runnable_sequence = prompt | llm  
        answer = runnable_sequence.invoke({"context": context, "question": user_input})

        print(f"Generated Answer: {answer}")

        session['chat_history'].append(f"You: {user_input}<br>AI: {answer}<br>")

        chat_history = "<br>".join(session['chat_history'])
        return jsonify({'response': chat_history})

    # Initialize chat history for GET requests
    session['chat_history'] = []
    return render_template_string(template)

if __name__ == '__main__':
    app.run(port=5000)
