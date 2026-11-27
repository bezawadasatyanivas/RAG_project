#reads in knowledge base 
#turns documents into chunks
#vectorize the chunks and stores in chroma

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = Path(__file__).parent.parent / "knowledge-base"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def fetch_documents():
    documents = []
    for folder in sorted(p for p in KNOWLEDGE_BASE.iterdir() if p.is_dir()):
        for file in sorted(folder.rglob("*.md")):
            documents.append(
                Document(
                    page_content=file.read_text(encoding="utf-8"),
                    metadata={"source": str(file), "doc_type": folder.name},
                )
            )
    print(f"Loaded {len(documents)} documents")
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Divided into {len(chunks)} chunks")
    return chunks


def create_embeddings(chunks):
    embeddings.embed_query("test")  # fail early, before touching the existing DB

    if Path(DB_NAME).exists():
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")