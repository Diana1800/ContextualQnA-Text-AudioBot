📱 Your content Q&A Bot with Text and Audio input Support 🤖

This is a Flask-based web application that allows users to ask questions via text input or upload audio files for transcription and receive answers using OpenAI's GPT-4 model. The app also leverages Whisper for audio transcription and supports multiple audio formats. The application logs the conversation history and provides feedback for unsupported file formats.

🧠 Key Features

Text-Based Q&A: Users can submit text questions, which are answered using OpenAI's GPT-4 model.
Audio Transcription: Users can upload audio files, which are transcribed into text using Whisper, and the transcribed text is used as the question input.
Supported Audio Formats: The app supports audio files in .wav, .mp3, .ogg, and .flac formats. It also converts .m4a files to .wav before transcription.
Conversation History: The app logs all questions and answers in the session, so users can scroll through the conversation history.
Error Handling: Graceful handling of unsupported file formats and missing inputs, with error messages returned to the user.

🛠️ Tech Stack

Flask: Backend framework for handling requests and rendering the front end.
OpenAI GPT-4: Used for generating answers to user questions.
Whisper: For transcribing audio input into text.
LangChain: Used for question-answering tasks with context management.
Pydub: Used for converting audio files (e.g., .m4a to .wav).
HTML/CSS/JavaScript: Basic front-end interface for interaction.

📦 Installation

Clone the repository:

    git clone https://github.com/diana1800/contextual-qna-bot.git
    cd contextual-qna-bot

Install the required packages from requirements.txt

Install Whisper from git:
    pip install git+https://github.com/openai/whisper.git

Update the env file with your OpenAI API key


Access the application: Open your web browser and go to http://127.0.0.1:5000.

⚙️ How It Works

Text Input: Users can enter their question in the text input field and press Go!. The question will be processed by GPT-4, and the response will be displayed below.
Audio Input: Users can upload an audio file by clicking the Choose File button, which is then transcribed using Whisper. The transcribed text is treated as the user input and processed by GPT-4.
History: Each interaction (both question and answer) is stored and displayed in a scrollable container.

🗂️ Project Structure


├── app.py                # Main Flask application

├── requirements.txt      # Python dependencies

├── templates             # HTML template

├── static                # Static files (CSS, JS)

└── README.md             # Project documentation

🖼️ User Interface

The app has a simple and clean user interface:

Top Section: Title and icon.
Middle Section: Scrollable area displaying conversation history (both text and audio).
Bottom Section: Input field for text and file upload button for audio. Users can submit questions or clear the audio file.

💡 How to Use

Submit a question:
Enter your question in the text box and click Go! to submit it.
Submit an audio file:
Upload an audio file (in .wav, .mp3, .ogg, .flac, or .m4a format), and the system will transcribe it and generate a response.
Clear audio file:
If you need to clear the selected audio file, click Clear to remove the file.




![image](https://github.com/user-attachments/assets/f83842a0-5719-4823-9489-b5035c947fd9)

