"""
rag_engine.py
─────────────
Core RAG logic: transcript fetching, chunking,
vector store building, and question answering.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str | None:
    """Extract YouTube video ID from URL or plain ID string."""
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    return None


def fetch_transcript(video_id: str) -> tuple[bool, str]:
    """
    Fetch English transcript for a YouTube video.
    Returns (success: bool, text_or_error: str)
    """
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=["en"])
        transcript = " ".join(snippet.text for snippet in fetched)
        return True, transcript
    except Exception as e:
        # Try auto-generated captions
        try:
            ytt = YouTubeTranscriptApi()
            fetched = ytt.fetch(video_id, languages=["en-US", "en-GB"])
            transcript = " ".join(snippet.text for snippet in fetched)
            return True, transcript
        except Exception as e2:
            return False, str(e2)


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class YouTubeRAG:
    """Full RAG pipeline for one or more YouTube videos."""

    def __init__(self):
        self.transcripts: dict[str, str] = {}   # video_id → raw transcript
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self._embeddings = None
        self._llm = None

    # ── Step 1: Add transcripts ───────────────────────────────────────────────
    def add_transcript(self, video_id: str, transcript: str):
        self.transcripts[video_id] = transcript

    def clear(self):
        self.transcripts.clear()
        self.vector_store = None
        self.retriever = None
        self.chain = None

    # ── Step 2: Build vector store ────────────────────────────────────────────
    def build(self, progress_callback=None):
        """
        Chunk all transcripts → embed → FAISS index.
        progress_callback(step: str) is called at each stage.
        """
        if not self.transcripts:
            raise ValueError("No transcripts loaded yet.")

        def _step(msg):
            if progress_callback:
                progress_callback(msg)

        _step("Splitting transcripts into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_text = "\n\n---VIDEO BOUNDARY---\n\n".join(
            f"[Video: {vid}]\n{txt}"
            for vid, txt in self.transcripts.items()
        )
        chunks = splitter.create_documents([all_text])

        _step("Loading embedding model (first run may take a moment)...")
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

        _step("Building FAISS vector store...")
        self.vector_store = FAISS.from_documents(chunks, self._embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _step("Connecting LLM...")
        if self._llm is None:
            self._llm = ChatGroq(model="llama-3.3-70b-versatile")

        _step("Pipeline ready!")
        self._build_chain()

    # ── Step 3: Build LangChain chain ─────────────────────────────────────────
    def _build_chain(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        parallel = RunnableParallel({
            "context": self.retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })

        # We store the parallel chain and build the full chain later
        # so we can inject the language at query time
        self._parallel = parallel
        self._parser = StrOutputParser()
        self.chain = True  # flag: built

    # ── Step 4: Ask a question ────────────────────────────────────────────────
    def ask(self, question: str, language: str = "English") -> str:
        """
        Ask a question. language = "English" | "Urdu"
        Returns the answer string.
        """
        if not self.chain:
            raise RuntimeError("Call build() first.")

        if language == "Urdu":
            lang_instruction = (
                "IMPORTANT: You MUST answer entirely in Urdu language "
                "using Urdu script (اردو). Do not mix English words unless "
                "they are technical terms with no Urdu equivalent."
            )
        else:
            lang_instruction = "Answer clearly in English."

        prompt = PromptTemplate(
            template="""You are YT Chat — an intelligent YouTube video assistant.
Answer ONLY from the provided transcript context below.
If the answer is not in the context, honestly say you don't know.
{lang_instruction}

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question", "lang_instruction"],
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            RunnableParallel({
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "lang_instruction": RunnableLambda(lambda _: lang_instruction),
            })
            | prompt
            | self._llm
            | self._parser
        )

        return chain.invoke(question)
