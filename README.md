📖 Full README.md
🚀 YouTube AI Chatbot (Multi-Video RAG)

An intelligent chatbot that allows you to interact with multiple YouTube videos at once. It extracts transcripts, processes them using embeddings, and enables you to ask questions with accurate, context-aware answers — just like ChatGPT, but focused on your selected videos.

🔥 Features
🎥 Multi-Video Support – Add multiple YouTube links
💬 Chat Interface – Ask natural language questions
🧠 RAG Pipeline – Uses Retrieval-Augmented Generation
⏱️ Timestamps – Answers include video time references
🔗 Source Attribution – Know exactly which video the answer came from
⚡ Fast Search – Vector database powered retrieval (FAISS/Chroma)
🤖 LLM Support – Works with Groq / DeepSeek / OpenAI models
🏗️ Tech Stack
LLM: Groq 
Embeddings: HuggingFace Embeddings
Vector DB: FAISS 
Framework: LangChain
Downloader: yt-dlp
Backend: Python
🧠 How It Works
Provide one or more YouTube video URLs
Extract transcripts using Whisper or YouTube captions
Split text into chunks
Convert text into embeddings
Store in vector database
Retrieve relevant chunks based on query
Generate answer using LLM
📂 Project Structure
youtube-ai-chatbot/
│── app.py
│── requirements.txt
│── utils/
│   ├── rag-engine.py
│── data/
│── vectorstore/
⚙️ Installation
git clone https://github.com/KashifAliChandio/youtube-ai-chatbot.git
cd youtube-ai-chatbot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
🔑 Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key
▶️ Usage
python app.py

Then:

Add YouTube video URLs
Ask questions like:
"Summarize the video"
"What did the speaker say about AI?"
"Explain topic at 5:30"
💡 Example Queries
“What are the key points from all videos?”
“Compare concepts explained in these videos”
“Give timestamps where neural networks are discussed”
📸 Demo (Optional)

Add screenshots or demo GIF here

🛠️ Future Improvements
🌐 Web UI (Streamlit)
🗂️ Save chat history
🎙️ Voice-based queries
📊 Video summarization dashboard
🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a PR.

📜 License

MIT License

⭐ Support

If you like this project, give it a ⭐ on GitHub!
