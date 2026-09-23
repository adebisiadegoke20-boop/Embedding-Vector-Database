# Knowledge Bot – Retrieval-Augmented Generation (RAG) System

## Project Overview
This project was developed as an assignment for the **NECA ICT Academy Advanced AI Automation Engineer Program**.

The project demonstrates how an AI-powered question-answering system can use a specific collection of documents as its knowledge base instead of relying solely on the general knowledge of a Large Language Model (LLM).

For this project, the knowledge base consists of **antenatal care documents**. Users can ask questions about the information contained in these documents, and the system retrieves relevant information before generating a response.

## Project Objective

The main objective of this project was to build a complete RAG system that can:

- Process and store information from multiple documents.
- Convert document content into vector embeddings.
- Retrieve relevant information based on a user's question.
- Use retrieved information as context for an LLM.
- Answer questions using the available knowledge base.
- Handle questions that require information from multiple documents.
- Recognize when a question cannot be answered from the available documents.
- Deploy the completed application using Streamlit.

## How the Knowledge Bot Works

The system follows a Retrieval-Augmented Generation workflow:

```text
User Question
      ↓
Question converted to Embedding
      ↓
Supabase Vector Database
      ↓
Relevant Document Information Retrieved
      ↓
Retrieved Context + User Question
      ↓
Groq LLM
      ↓
Generated Answer
      ↓
User
```

First, the documents are converted into numerical embeddings using the **all-MiniLM-L6-v2** model. These embeddings are stored in **Supabase**, which is used as the vector database.

When a user asks a question, the question is also converted into an embedding. The system then performs a similarity search to retrieve the most relevant information from the stored documents.

The retrieved information is provided to the **Groq LLM** as context, allowing it to generate an answer based on the project's knowledge base.

## Documents Used

The Knowledge Bot was developed using documents related to **antenatal care**. These documents provide the information that the system retrieves when answering user questions.

The system was designed to work with multiple documents so that questions requiring information from different sources can also be tested.

## Technologies Used

- **Python** – Application development
- **Streamlit** – Web interface and deployment
- **Supabase** – Vector database and similarity search
- **Sentence Transformers** – Text embedding
- **all-MiniLM-L6-v2** – Embedding model
- **Groq** – Large Language Model
- **python-dotenv** – Environment variable management

## Testing

The Knowledge Bot was tested using different types of questions:

### 1. Questions Based on the Documents

Questions were asked about information contained in the antenatal care documents to verify that the system could retrieve relevant information and generate appropriate responses.

### 2. Multi-Document Questions

Questions requiring information from more than one document were used to test the system's ability to retrieve relevant information across multiple sources.

### 3. Questions Outside the Knowledge Base

Questions that were not covered by the available documents were also tested.

The purpose of this test was to ensure that the system could recognize when the required information was not available instead of generating an unsupported response.

## Project Structure

```text
Embedding-Vector-Database/
│
├── app.py
├── Video11-3.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Live Application
The completed Knowledge Bot is deployed using Streamlit and can be accessed here:
**[Open Knowledge Bot](https://embedding-vector-database-jax4happtarjkq3uvqshdtc.streamlit.app/)**

## Conclusion
This project demonstrates the practical implementation of a Retrieval-Augmented Generation system, combining document embeddings, vector search, and Large Language Model generation.

The completed Knowledge Bot provides a simple interface for asking questions about antenatal care documents and demonstrates how RAG can be used to generate responses grounded in a specific knowledge base.
