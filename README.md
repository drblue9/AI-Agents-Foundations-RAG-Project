# Meeting Notes Q&A & Summarizer

A RAG-based app for meeting transcripts and notes. Upload one or more
transcripts, then:

- **Summarize** any transcript into decisions, action items, and open
  questions.
- **Ask questions** across all ingested transcripts and get answers
  grounded only in what was actually discussed.

Built on the same stack as the original RAG Document Q&A project:

- **Streamlit** — UI
- **sentence-transformers (`all-MiniLM-L6-v2`)** — local embeddings
- **ChromaDB** — in-memory vector store
- **Gemini** (via `google-genai`) — answer generation & summarization

## What's different from the original

- Supports multiple files ingested into one shared collection (so you
  can ask questions across several meetings at once), instead of
  resetting the collection on every upload.
- Adds `.vtt` / `.srt` transcript export formats alongside PDF/TXT/DOCX.
- Adds a dedicated **Summarize** step that produces a structured
  Markdown summary (Key Decisions / Action Items / Open Questions)
  rather than only Q&A.
- Tracks ingested files in session state so you can pick which one to
  summarize, and clear them from the sidebar.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:

   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-3.6-flash
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload one or more meeting transcripts (PDF, TXT, DOCX, VTT, or
   SRT) and click **Ingest Transcript(s)**.
2. Pick a transcript and click **Generate Summary** for a structured
   recap.
3. Ask a question in plain English — it searches across every
   ingested meeting and answers using only the retrieved context.

A sample transcript is included at `data/sample_meeting.txt` for
testing.
