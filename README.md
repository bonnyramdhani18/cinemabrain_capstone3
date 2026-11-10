# CinemaBrain - IMDb Movie Recommender & Assistant

Aplikasi rekomendasi film IMDb berbasis Agentic RAG menggunakan LangGraph, Qdrant Vector Database, dan Streamlit.

## Live Demo
Link Streamlit Cloud(https://cinemabrain.streamlit.app)

## Fitur Utama
- Agentic RAG: Menggunakan LangGraph (ReAct agent) untuk memutuskan kapan harus mengambil data dari database vektor secara dinamis.
- Semantic Search: Vector store Qdrant Cloud diindeks menggunakan model embedding `sentence-transformers/all-MiniLM-L6-v2`.
- Conversational Memory: Menjaga konteks percakapan.
- Observability: Menampilkan tool calls dan detail penggunaan token LLM (input/output).
- Quick Prompts: Saran pertanyaan.