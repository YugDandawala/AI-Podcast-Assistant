# 🎙️ AI Podcast Assistant: YouTube RAG Chatbot

The **AI Podcast Assistant** is a sophisticated Retrieval-Augmented Generation (RAG) system built to allow users to chat directly with the content of any YouTube podcast. By fetching the video's transcript, indexing it, and combining it with a powerful Large Language Model (LLM), the assistant provides context-aware, domain-specific answers.

The assistant is strictly scoped to its knowledge base, meaning it will refuse to answer any questions unrelated to the podcast content it has been loaded with.

## 🚀 Key Features

*   **YouTube Transcript Extraction:** Uses the robust `yt-dlp` library to reliably fetch English and Hindi subtitles/captions from any public YouTube video.
*   **Retrieval-Augmented Generation (RAG):** Implements a full RAG pipeline using LangChain for grounded, accurate answers.
*   **Domain-Specific Persona:** Employs a strict system prompt to enforce a "PodcastScope" persona, ensuring the assistant maintains focus and refuses off-topic questions.
*   **Streamlit UI:** Provides a clean, interactive web interface for loading transcripts and engaging in conversation.
*   **Persistent Vector Store:** Utilizes **PostgreSQL with PGVector** for scalable and persistent storage of vector embeddings.

## 🛠️ Technical Stack

| Component | Technology/Library |
| :--- | :--- |
| **Application Framework** | Python / Streamlit |
| **LLM Orchestration** | LangChain |
| **LLM Model** | Llama 3.1 8B Instruct (via Hugging Face Endpoint) |
| **Embedding Model** | `all-MiniLM-L6-v2` (via Hugging Face Endpoint) |
| **Vector Database** | PostgreSQL with PGVector |
| **Transcript Source** | `yt-dlp` |
| **Configuration** | `python-dotenv` |

## ⚙️ Setup and Installation

### 1. Prerequisites

You must have the following installed:

*   **Python 3.x**
*   **PostgreSQL** (with the `pgvector` extension enabled on your database instance).

### 2. Install Dependencies

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/YugDandawala/AI-Podcast-Assistant.git
cd AI-Podcast-Assistant
pip install -r requirements.txt
```

### 3. Database Configuration

The application is configured to connect to a local PostgreSQL instance.

**⚠️ Security Warning:** The current scripts (`other.py` and `three.py`) contain hardcoded database credentials (`user="postgres"`, `password="Admin@123"`). **It is highly recommended to replace these with environment variables.**

1.  Start your PostgreSQL server.
2.  Create the required database:
    ```sql
    CREATE DATABASE langchain;
    ```
3.  Ensure your user/password matches the hardcoded values or modify the connection string in the Python script (`other.py`).

### 4. API Key Configuration

The application requires an API key for the Hugging Face Endpoint.

1.  Create a file named `.env` in the root of your project directory.
2.  Add your Hugging Face API key:

    ```
    HUGGINGFACEHUB_API_TOKEN="hf_YOUR_API_KEY_HERE"
    ```

### 5. Run the Application

Execute the main script (`other.py`) using Streamlit:

```bash
streamlit run other.py
```

The application will open in your web browser. Enter a YouTube video ID (e.g., `dQw4w9WgXcQ`), click **"Load Podcast Transcript"**, and begin asking questions about the video's content.

## 📝 Project Details

*   **Main Script:** `other.py`
*   **Database Connection:** `postgresql+psycopg2://postgres:Admin@123@localhost:5432/langchain`
*   **Collection Name:** `embedded_vectors`
*   **Text Splitter:** `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200)