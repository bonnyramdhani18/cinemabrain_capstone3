import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import toml

#Konfig secrets.toml
secrets = toml.load(".streamlit/secrets.toml")
QDRANT_URL = secrets["QDRANT_URL"]
QDRANT_API_KEY = secrets["QDRANT_API_KEY"]
COLLECTION_NAME = "imdb_movies"
print("1. Membaca dataset IMDb...")

df = pd.read_csv("imdb_top_1000.csv").fillna("")

#Ambil 300 data pertama imdb
df = df.head(300)

print("2. Menyiapkan data dokumen...")
documents = []
for _, row in df.iterrows():
    content = (
        f"Title: {row['Series_Title']}\n"
        f"Genre: {row['Genre']}\n"
        f"Director: {row['Director']}\n"
        f"Stars: {row['Star1']}, {row['Star2']}\n"
        f"Rating: {row['IMDB_Rating']}\n"
        f"Overview: {row['Overview']}"
    )
    metadata = {
        "title": str(row["Series_Title"]),
        "genre": str(row["Genre"]),
        "rating": str(row["IMDB_Rating"]),
        "year": str(row["Released_Year"])
    }
    documents.append(Document(page_content=content, metadata=metadata))

print("3. Inisialisasi Model Embedding (HuggingFace)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("4. Menghubungkan ke Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

#Buat collection baru
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' berhasil dibuat.")

print("5. Mengunggah vektor ke Qdrant Cloud (tunggu sebentar)...")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
vector_store.add_documents(documents=documents)
print(" Selesai! Data film berhasil tersimpan di Qdrant Cloud.")