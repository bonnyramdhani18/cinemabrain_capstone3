import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

# Ambil kredensial dari st.secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Inisialisasi LLM via OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Embedding (sama persis dengan ingest.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

collection_name = "imdb_movies"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

@tool
def get_relevant_movies(question: str):
    """Use this tool to search and retrieve relevant movie information, plots, ratings, and genres from the IMDb database."""
    results = qdrant.similarity_search(question, k=5)
    return results

tools = [get_relevant_movies]

def chat_imdb(question, history):
    system_prompt = (
        "You are an expert movie assistant. Answer questions about movies, recommendations, genres, "
        "and plots using the provided movie database tools. Always check relevant movies through the tool before answering.\n"
        f"Conversation History:\n{history}"
    )
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if hasattr(message, "response_metadata"):
            usage = message.response_metadata.get("usage_metadata") or message.response_metadata.get("token_usage", {})
            total_input_tokens += usage.get("input_tokens") or usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("output_tokens") or usage.get("completion_tokens", 0)

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_messages.append(message.content)

    return {
        "answer": answer,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }

st.set_page_config(page_title="CinemaBrain", page_icon="🎬")
st.title("🎬 CinemaBrain")

# Inisialisasi session state chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick prompt suggestion
prompt = None
if len(st.session_state.messages) == 0:
    st.write("💡 **Coba tanyakan contoh ini:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🦇 Sinopsis Batman Begins"):
            prompt = "Berikan sinopsis film Batman Begins"
    with col2:
        if st.button("🚀 Film Sci-Fi Rating Tertinggi"):
            prompt = "Rekomendasikan film Sci-Fi dengan rating tertinggi"
    with col3:
        if st.button("🎭 Rekomendasi Film Drama"):
            prompt = "Berikan rekomendasi film drama terbaik"

# Tampilkan history obrolan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Form input pengguna
if prompt := st.chat_input("Tanyakan rekomendasi film, sinopsis, atau genre..."):
    messages_history = st.session_state.get("messages", [])[-6:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or "None"

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = chat_imdb(prompt, history)
        st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    with st.expander("**Tool Calls (RAG Document Retrieval):**"):
        st.code(response["tool_messages"])

    with st.expander("**History Chat:**"):
        st.code(history)

    with st.expander("**Usage Details:**"):
        st.code(f'Input Tokens  : {response["total_input_tokens"]}\nOutput Tokens : {response["total_output_tokens"]}')