import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Knowledge Bot",
    page_icon="",
    layout="wide"
)

# Initialize Connections
def innit_connection():
    supabase = create_client(os.environ.get("SUPABASE_URL"),os.environ.get("SUPABASE_SERVICE_KEY"))
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return supabase, embedding_model, groq_client
supabase, embedding_model, groq_client = innit_connection()

# Document Processing
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1

        chunk = text[start:end].strip()

        if len(chunk) > 20:
            chunks.append(chunk)
        new_start = end - overlap
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks


# Ingest Documents
def ingest_document(title, content, source):
    chunks = chunk_text(content)
    stored = 0
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        supabase.table("documents").insert({
            "title": title,
            "content": chunk,
            "source": source,
            "page_number": i + 1,
            "embedding": embedding
        }).execute()
        stored += 1
    return stored


# RAG Search
def rag_search(question, top_k=5):
    query_embedding = embedding_model.encode(question).tolist()
    results = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.15,
            "match_count": top_k
        }
    ).execute()
    return results.data


# RAG Generate
def rag_generate(question, documents):
    context = ""

    for i, doc in enumerate(documents):
        context += "Source: " + doc["title"] + " (chunk " + str(doc["page_number"]) + ")\n"
        context += doc["content"]

        if i < len(documents) - 1:
            context += "\n---\n"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """Answer questions based ONLY on the provided documents.
If the answer cannot be found in the documents, say:
"I don't have enough information in the provided documents."
Do not make up information.
Cite sources using [Source: Document Title]."""
            },
            {
                "role": "user",
                "content": "Documents:\n" + context + "\n\nQuestion: " + question
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# Sidebar - Document Upload
with st.sidebar:
    st.header("Add Documents")
    uploaded_file = st.file_uploader("Upload a text file",type=["txt"])
    doc_title = st.text_input("Document title")

    if st.button("Ingest Document"):
        if uploaded_file and doc_title:
            content = uploaded_file.read().decode("utf-8")
            with st.spinner("Chunking, embedding, and storing..."):
                count = ingest_document(doc_title,content,uploaded_file.name)
        else:
            st.warning("Please provide both a file and a title.")
    try:
        count = supabase.table("documents").select("id",count="exact").execute()
        st.caption("Documents in database: " + str(count.count))
    except:
        st.caption("Database connection error")


# Main Chat Interface
st.title("Knowledge Bot")
st.caption("Ask questions about the documents in the knowledge base.")


# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write("- " + s["title"] + " (similarity: " +str(s["similarity"]) + ")")


# Chat Input
question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user","content": question})
    with st.chat_message("user"):
        st.markdown(question)

# Generate answers
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            docs = rag_search(question)

            if not docs:
                answer = "I don't have enough information in the provided documents."
                sources = []
            else:
                answer = rag_generate(question, docs)
                sources = []
                for doc in docs:
                    sources.append({"title": doc["title"],"similarity": round(doc["similarity"], 3)})
        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write("- " + s["title"] +" (similarity: " +str(s["similarity"]) + ")")
    