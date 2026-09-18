import os
import tempfile

import streamlit as st
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env file.")
    st.stop()


# Gemini client
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

LLM_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

# Local embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Meeting Notes Q&A & Summarizer",
    page_icon="🗒️",
    layout="wide"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.Client()

COLLECTION_NAME = "meeting_notes"

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# HELPER: EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(file_path, file_type):

    # PDF
    if file_type == "pdf":

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    # TXT / VTT / SRT (plain-text style transcript exports)
    elif file_type in ("txt", "vtt", "srt"):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    # DOCX
    elif file_type == "docx":

        document = Document(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

    return ""


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(text):

    chunks = []

    start = 0

    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# ============================================================
# TRANSCRIPT INGESTION (supports multiple meeting files)
# ============================================================

def ingest_transcript(uploaded_file, reset_collection):

    file_extension = (
        uploaded_file.name
        .split(".")[-1]
        .lower()
    )

    # --------------------------------------------------------
    # Save uploaded file temporarily
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{file_extension}"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name

    try:

        text = extract_text(
            temp_path,
            file_extension
        )

    finally:

        os.remove(temp_path)

    if not text.strip():

        st.error(
            f"Could not extract text from {uploaded_file.name}."
        )

        return 0, text

    # --------------------------------------------------------
    # Create chunks + embeddings
    # --------------------------------------------------------

    chunks = create_chunks(text)

    embeddings = create_embeddings(chunks)

    global collection

    if reset_collection:

        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    existing_count = collection.count()

    ids = [
        f"{uploaded_file.name}_chunk_{existing_count + i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": uploaded_file.name,
            "chunk": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks), text


# ============================================================
# SIMILARITY SEARCH
# ============================================================

def search_transcripts(question):

    question_embedding = create_embeddings(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, max(collection.count(), 1))
    )

    return results


# ============================================================
# GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an assistant that answers questions about meeting
transcripts and notes.

Your task is to answer the user's question using ONLY the
information provided in the retrieved meeting context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information, decisions, or names.
3. If the answer is not present in the context, say exactly:

   "That wasn't discussed in the meeting notes provided."

4. Where relevant, mention who said or owns something, if the
   context makes that clear.
5. Give a clear and concise answer.

--------------------------------------------------
RETRIEVED MEETING CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
ANSWER
--------------------------------------------------
"""

    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text


# ============================================================
# GENERATE STRUCTURED MEETING SUMMARY
# ============================================================

def generate_summary(full_text):

    # Guard against extremely long transcripts blowing past
    # context limits — summarize from the first N characters
    # of raw text plus a note if truncated.
    MAX_CHARS = 30000

    truncated = full_text[:MAX_CHARS]

    truncation_note = (
        "\n\n[Note: transcript truncated for summarization]"
        if len(full_text) > MAX_CHARS else ""
    )

    prompt = f"""
You are an assistant that writes structured meeting summaries.

Read the meeting transcript below and produce a summary with
these sections, using Markdown headings:

## Summary
A short 2-4 sentence overview of what the meeting was about.

## Key Decisions
Bullet points of decisions that were made. Say "None recorded"
if there were none.

## Action Items
Bullet points in the form "- [Owner, if known]: Task". Say
"None recorded" if there were none.

## Open Questions / Follow-ups
Bullet points of unresolved questions or things to follow up
on. Say "None recorded" if there were none.

Base everything strictly on the transcript. Do not invent
names, owners, or facts that are not present in the text.

--------------------------------------------------
TRANSCRIPT
--------------------------------------------------

{truncated}{truncation_note}
"""

    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text


# ============================================================
# SESSION STATE
# ============================================================

if "ingested_texts" not in st.session_state:
    st.session_state.ingested_texts = {}


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🗒️ Meeting Notes Q&A & Summarizer")

st.write(
    "Upload one or more meeting transcripts or notes. Ask "
    "questions across them, or generate a structured summary "
    "with decisions, action items, and follow-ups."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write(f"**Chunk Size:** {CHUNK_SIZE}")
    st.write(f"**Chunk Overlap:** {CHUNK_OVERLAP}")
    st.write(f"**Top K:** {TOP_K}")
    st.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
    st.write(f"**LLM:** {LLM_MODEL}")
    st.write("**Vector Database:** ChromaDB")

    st.divider()

    st.write("**Ingested Transcripts:**")

    if st.session_state.ingested_texts:
        for name in st.session_state.ingested_texts:
            st.write(f"- {name}")
    else:
        st.caption("None yet.")

    if st.button("🗑️ Clear All Transcripts"):

        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        st.session_state.ingested_texts = {}
        st.rerun()


# ============================================================
# STEP 1: UPLOAD TRANSCRIPT(S)
# ============================================================

st.header("1️⃣ Upload Meeting Notes")

uploaded_files = st.file_uploader(
    "Upload PDF, TXT, DOCX, VTT or SRT transcripts",
    type=["pdf", "txt", "docx", "vtt", "srt"],
    accept_multiple_files=True
)

if uploaded_files:

    if st.button("🚀 Ingest Transcript(s)"):

        with st.spinner(
            "Extracting text, chunking, embedding and "
            "indexing..."
        ):

            total_chunks = 0

            for uploaded_file in uploaded_files:

                try:

                    n_chunks, full_text = ingest_transcript(
                        uploaded_file,
                        reset_collection=False
                    )

                    if n_chunks > 0:

                        st.session_state.ingested_texts[
                            uploaded_file.name
                        ] = full_text

                        total_chunks += n_chunks

                except Exception as e:

                    st.error(
                        f"Error ingesting {uploaded_file.name}: {e}"
                    )

            if total_chunks > 0:

                st.success(
                    f"Indexed {total_chunks} chunks across "
                    f"{len(uploaded_files)} file(s)."
                )


# ============================================================
# STEP 2: SUMMARIZE
# ============================================================

st.header("2️⃣ Summarize a Meeting")

if st.session_state.ingested_texts:

    selected_doc = st.selectbox(
        "Choose a transcript to summarize",
        options=list(st.session_state.ingested_texts.keys())
    )

    if st.button("📝 Generate Summary"):

        with st.spinner("Generating summary using Gemini..."):

            try:

                summary = generate_summary(
                    st.session_state.ingested_texts[selected_doc]
                )

                st.markdown(summary)

            except Exception as e:

                st.error(f"Error generating summary: {e}")

else:

    st.caption("Upload and ingest a transcript first.")


# ============================================================
# STEP 3: ASK A QUESTION
# ============================================================

st.header("3️⃣ Ask a Question")

question = st.text_input(
    "Ask about any ingested meeting (e.g. \"What did we decide "
    "about the launch date?\")"
)

if st.button("🔍 Ask Question"):

    if not question:

        st.warning("Please enter a question.")

    elif collection.count() == 0:

        st.warning("Please upload and ingest a transcript first.")

    else:

        with st.spinner("Searching meeting notes..."):

            try:

                results = search_transcripts(question)

                retrieved_chunks = results["documents"][0]
                distances = results["distances"][0]

            except Exception as e:

                st.error(f"Error during retrieval: {e}")
                st.stop()

        with st.spinner("Generating answer using Gemini..."):

            try:

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            except Exception as e:

                st.error(f"Error generating answer: {e}")
                st.stop()

        st.subheader("💡 Answer")
        st.success(answer)

        st.subheader("📄 Retrieved Context")

        for i, chunk in enumerate(retrieved_chunks):

            with st.expander(f"Retrieved Chunk {i + 1}"):

                st.write(chunk)

                metadata = results["metadatas"][0][i]

                st.caption(
                    f"Source: {metadata['source']} | "
                    f"Chunk: {metadata['chunk']} | "
                    f"Distance: {distances[i]:.4f}"
                )
