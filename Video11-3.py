from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)

print("System Initialized")
print("Connected to Supabase")
print("Embedding Model Loaded")
print("Groq Client Ready")

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
    print()
    print("Ingesting:", title)
    print("-" * 40)

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

    print("Stored", stored, "chunks from", title)
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
        context += (
            "Source: " + doc["title"] +
            " (chunk " + str(doc["page_number"]) + ")\n"
        )
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


# Full RAG
def full_rag(question):
    print()
    print("Question:", question)
    print("-" * 50)

    print("Searching for relevant documents...")
    docs = rag_search(question)

    if not docs:
        print("No relevant documents found.")
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": []
        }

    print("Found", len(docs), "relevant chunks")

    for doc in docs:
        print(
            "-",
            doc["title"],
            "(similarity:",
            round(doc["similarity"], 3),
            ")"
        )

    print()
    print("Generating answer...")
    answer = rag_generate(question, docs)

    sources = []

    for doc in docs:
        sources.append({
            "title": doc["title"],
            "chunk": doc["page_number"],
            "similarity": round(doc["similarity"], 3)
        })

    print()
    print("Answer:")
    print(answer)

    print()
    print("Sources:")

    for source in sources:
        print(
            "-",
            source["title"],
            "(chunk",
            source["chunk"],
            ")"
        )

    return {
        "answer": answer,
        "sources": sources
    }


# Test Documents
print()
print("Storing documents")
print("-" * 50)

antenatal_content = """
Antenatal Care Guide: Pregnant patients are encouraged to begin antenatal care as early as possible after confirming pregnancy.
Routine antenatal visits may include checking blood pressure, monitoring weight, reviewing symptoms, and discussing the patient's pregnancy history.
Patients should follow the appointment schedule provided by their healthcare provider.
Pregnant patients should inform their healthcare provider about any medications they are taking.
"""

warning_signs_content = """
Pregnancy Warning Signs Guide: During pregnancy, symptoms such as heavy vaginal bleeding, severe abdominal pain, difficulty breathing, loss of consciousness, or severe headache with vision changes may require urgent medical attention.
Patients experiencing severe or rapidly worsening symptoms should seek immediate medical care rather than waiting for a routine appointment.
This document is for educational purposes and does not replace assessment by a qualified healthcare professional.
"""

appointment_content = """
Antenatal Appointment Policy: Patients can schedule antenatal appointments with the maternity clinic.
Patients should provide their name, contact information, expected delivery date if known, and reason for the appointment.
Routine antenatal appointments are intended for pregnancy monitoring and follow-up.
Patients with urgent symptoms should seek appropriate medical attention instead of waiting for a routine appointment.
"""

nutrition_content = """
Pregnancy Nutrition Guide: A balanced diet during pregnancy can include vegetables, fruits, whole grains, protein sources, and adequate fluids.
Pregnant patients should discuss supplements such as iron and folic acid with their healthcare provider.
Food choices and nutritional needs may vary between individuals, so patients should seek personalized advice from a qualified healthcare professional.
"""


ingest_document(
    "Antenatal Care Guide",
    antenatal_content,
    "antenatal_care.txt"
)

ingest_document(
    "Pregnancy Warning Signs Guide",
    warning_signs_content,
    "warning_signs.txt"
)

ingest_document(
    "Antenatal Appointment Policy",
    appointment_content,
    "antenatal_appointments.txt"
)

ingest_document(
    "Pregnancy Nutrition Guide",
    nutrition_content,
    "pregnancy_nutrition.txt"
)


# Tests
print()
print("=" * 50)
print("RAG TESTS")
print("=" * 50)

full_rag(
    "When should a pregnant patient begin antenatal care "
    "and what can be checked during routine visits?"
)

full_rag(
    "What symptoms during pregnancy may require urgent medical attention?"
)

full_rag(
    "What information should a patient provide when booking "
    "an antenatal appointment?"
)

full_rag(
    "A pregnant patient wants to attend an antenatal appointment "
    "and is also asking about nutrition and supplements. "
    "What information is available in the documents?"
)

full_rag(
    "A pregnant patient has severe abdominal pain. "
    "Should they wait for their routine antenatal appointment?"
    )

full_rag("What hospital in Abuja provides free antenatal care?")