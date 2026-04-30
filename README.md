# YT Chat — YouTube AI Chatbot
### Streamlit + LangChain + Groq + FAISS

---

## 📁 File Structure (Folder ka Haal)

```
yt_chatbot/
├── app.py              ← Main Streamlit app (yahi chalao)
├── rag_engine.py       ← RAG pipeline logic (transcript + FAISS + LLM)
├── requirements.txt    ← Saari libraries ki list
├── .env                ← API keys (GROQ_API_KEY yahan daalo)
└── README.md           ← Yeh file
```

---

## ⚡ Setup — Step by Step

### Step 1 — Python version check karo
```bash
python --version
# Python 3.10 ya usse upar hona chahiye
```

### Step 2 — Virtual environment banao (zaroori hai!)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Libraries install karo
```bash
pip install -r requirements.txt
```
> ⚠️ Pehli baar thoda waqt lagega (sentence-transformers bada hai ~500MB)

### Step 4 — API Key set karo
`.env` file kholein aur apni Groq key daalen:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```
> 🔑 Free key milegi: https://console.groq.com/keys

### Step 5 — App chalao!
```bash
streamlit run app.py
```
Browser mein khulega: **http://localhost:8501**

---

## 🎮 Kaise Use Karo

1. **Left sidebar** mein YouTube link ya Video ID daalo
   - Example: `https://www.youtube.com/watch?v=UabBYexBD4k`
   - Ya sirf ID: `UabBYexBD4k`
2. **"+ Add Video"** dabao — ek ya zyada videos add kar sakte ho
3. Language chunno: **English** ya **Urdu — اردو**
4. **"⚡ Load & Analyze Videos"** dabao
5. Loading khatam hone ke baad chat mein sawal poochho!

---

## 🛠 Har Library Kya Karti Hai

| Library | Kaam |
|---------|------|
| `streamlit` | Website / UI banana |
| `langchain` | RAG pipeline ka framework |
| `langchain-groq` | Groq (LLaMA model) se connect karta hai |
| `langchain-huggingface` | HuggingFace embeddings load karta hai |
| `langchain-community` | FAISS vector store support |
| `youtube-transcript-api` | YouTube se transcript download karta hai |
| `faiss-cpu` | Fast similarity search (CPU version) |
| `sentence-transformers` | Text ko numbers (vectors) mein badalta hai |
| `python-dotenv` | .env file se API key padhta hai |

---

## ❓ Common Errors aur Fix

| Error | Fix |
|-------|-----|
| `GROQ_API_KEY not found` | `.env` mein key daalo |
| `No transcript available` | Video mein English captions hone chahiye |
| `ModuleNotFoundError` | `pip install -r requirements.txt` dobara chalao |
| `AttributeError: get_transcript` | `youtube-transcript-api>=1.0.0` chahiye — naya version |
| Port already in use | `streamlit run app.py --server.port 8502` |

---

## 🌟 Features

- ✅ Multi-video support (ek se zyada videos ek saath)
- ✅ English aur Urdu mein jawab
- ✅ Chat history (pichle sawaal yaad hain)
- ✅ LLaMA 3.3 70B model (Groq ke through — free!)
- ✅ FAISS vector search (fast similarity)
- ✅ Beautiful dark UI

---

## 🔒 Security Note
`.env` file ko **kabhi bhi** GitHub pe push mat karo!
`.gitignore` mein yeh line zaror daalo:
```
.env
venv/
__pycache__/
```
