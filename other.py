import streamlit as st
import yt_dlp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_postgres import PGVector
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import quote_plus
from dotenv import load_dotenv
import time
import re

load_dotenv()

st.set_page_config(
    page_title="Podcast AI Assistant",
    layout="wide",
    page_icon="🎙️"
)

st.title("🎙️ Podcast AI Assistant")
st.write("Upload a YouTube podcast ID and ask questions about it!")

st.sidebar.header("📥 Podcast Input")

video_id = st.sidebar.text_input(
    "YouTube Podcast ID",
    placeholder="Enter video ID..."
)

load_button = st.sidebar.button("Load Podcast Transcript")

# -----------------------------
# DATABASE + MODEL CONFIG
# -----------------------------
user = "postgres"
password = quote_plus("Admin@123")
host = "localhost"
database = "langchain"
collection = "embedded_vectors"

connectDB = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{database}"

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="chat-completion",
    temperature=0.6
)

model = ChatHuggingFace(llm=llm)

# SYSTEM PROMPT
SYSTEM_PROMPT = SystemMessage(content="""
You are PodcastScope, an AI Assistant whose only purpose is to provide information, analysis, discussion, and support exclusively related to the podcast.You must not answer or engage with any question, request, or topic that is not explicitly and directly related to this podcast. 1. Allowed Content (ONLY):You may answer ONLY if the user's request is about:Episodes (summaries, themes, topics, discussions)Guests (their role, quotes, contributions in the podcast)Storylines, arcs, motifs, or recurring themesBehind-the-scenes information directly related to the podcastRelease schedules, production details, or podcast creatorsInterpretation or analysis of podcast contentIf the content is not directly tied to the podcast, you must refuse. 2. Mandatory Refusal for Off-Topic Requests:For ANY question unrelated to the podcast, respond with this exact sentence, and do not modify or expand it:I'm sorry, but I can only provide information and assistance related to podcast. I do not have access to answer questions outside this scope. Do NOT attempt to answer the question.Do NOT provide partial information.Do NOT provide alternative suggestions outside the podcast domain.Do NOT create analogies or fictional links.3. If the User Insists or Repeats an Off-Topic Request:Use this strict follow-up response (exact wording):As I mentioned, I can only provide assistance related to Podcast . Let's focus on episodes, guests, or themes from the podcast.4. Behavior Requirements:Never break scope.Never speculate outside podcast-related material.Never provide general knowledge, current events, politics, science, definitions, advice, or any non-podcast content.Maintain a professional, friendly, and engaging tone at all times.When appropriate, guide the user back to discussing episodes, themes, or guests from the podcast. 5. Examples of Correct Behavior:User: "What was discussed in episode 5?"You: Provide a detailed, accurate summary, themes, and guest information.User: "Who is the president of the United States?"You:I'm sorry, but I can only provide information and assistance related to Podcast. I do not have access to answer questions outside this scope.User: "Come on, just tell me general stuff, please."You:As I mentioned, I can only provide assistance related to Podcast . Let's focus on episodes, guests, or themes from the podcast.This is your permanent identity and operational scope. You must never break these rules.
""")

# -----------------------------
# SESSION STATE
# -----------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SYSTEM_PROMPT]

# -----------------------------
# YT-DLP TRANSCRIPT EXTRACTION FUNCTION
# -----------------------------
def extract_transcript_with_ytdlp(video_id):
    """Extract transcript using yt-dlp - works with manual and auto-generated subtitles"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'hi'],
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info
            info = ydl.extract_info(url, download=False)
            
            # Check for subtitles or automatic captions
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})
            
            # Priority: English manual > English auto > Hindi manual > Hindi auto
            langs = ['en', 'hi']
            for lang in langs:
                if lang in subtitles and subtitles[lang]:
                    subtitle_url = subtitles[lang][0]['url']
                    return fetch_vtt_content(subtitle_url)
                elif lang in automatic_captions and automatic_captions[lang]:
                    subtitle_url = automatic_captions[lang][0]['url']
                    return fetch_vtt_content(subtitle_url)
            
            raise Exception("No subtitles found in English or Hindi")
            
    except Exception as e:
        st.error(f"Error extracting transcript: {str(e)}")
        return None

def fetch_vtt_content(subtitle_url):
    """Fetch and parse VTT subtitle content to plain text"""
    import requests
    try:
        response = requests.get(subtitle_url, timeout=10)
        if response.status_code == 200:
            vtt_content = response.text
            
            # Simple VTT parser - extract text content
            lines = vtt_content.split('\n')
            transcript_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip WEBVTT header, timestamps, and empty lines
                if (line.startswith('WEBVTT') or 
                    '-->' in line or 
                    line == '' or 
                    line.isdigit() or
                    line.startswith('NOTE')):
                    continue
                if line:
                    transcript_lines.append(line)
            
            return ' '.join(transcript_lines)
    except:
        return None
    return None

# -----------------------------
# LOAD & PROCESS TRANSCRIPT
# -----------------------------
if load_button:
    if video_id.strip() == "":
        st.warning("Please enter a YouTube video ID.")
        st.stop()

    with st.spinner("🔄 Fetching transcript with yt-dlp..."):
        transcript = extract_transcript_with_ytdlp(video_id)
        
        if not transcript or len(transcript.strip()) < 100:
            st.error("❌ No transcript found. The video may not have English/Hindi subtitles enabled.")
            st.info("💡 Try videos with 'CC' (closed captions) enabled.")
            st.stop()

    st.success(f"✅ Transcript loaded! ({len(transcript.split())} words)")

    with st.spinner("✂️ Splitting text into chunks..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

    with st.spinner("🧠 Creating embeddings and vector database..."):
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = PGVector.from_documents(
            embedding=embeddings,
            documents=chunks,
            collection_name=collection,
            connection=connectDB
        )

        st.session_state.retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    st.success("🎉 Podcast AI ready! Ask questions below.")

# -----------------------------
# CHAT AREA
# -----------------------------
st.subheader("💬 Ask Questions About This Podcast")

query = st.text_input("Type your question:", placeholder="Ask anything about the podcast...")

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def build_prompt(context, history, question):
    text = f"""Context from podcast transcript:
{context}

Previous conversation:
"""
    
    for msg in history[1:]:  # Skip system prompt
        if isinstance(msg, HumanMessage):
            text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            text += f"Assistant: {msg.content}\n"

    text += f"\nCurrent question: {question}\nAssistant: "
    return text

def run_query(question):
    retriever = st.session_state.retriever
    if retriever is None:
        return "⚠️ Please load a podcast transcript first by entering a video ID and clicking 'Load Podcast Transcript'."

    try:
        docs = retriever.invoke(question)
        context = format_docs(docs)

        prompt = build_prompt(context, st.session_state.chat_history, question)
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Handle query submission
if query:
    st.session_state.chat_history.append(HumanMessage(content=query))

    with st.spinner("🤖 Generating answer..."):
        answer = run_query(query)

    st.session_state.chat_history.append(AIMessage(content=answer))
    
    # Auto-scroll to bottom
    st.rerun()

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
st.subheader("📜 Conversation History")

chat_container = st.container()

with chat_container:
    for i, msg in enumerate(st.session_state.chat_history):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

# -----------------------------
# INFO & REQUIREMENTS
# -----------------------------
with st.expander("ℹ️ Requirements & Setup"):
    st.markdown("""
    ### 📦 Install Dependencies
    ```
    pip install streamlit yt-dlp langchain langchain-huggingface langchain-postgres psycopg2-binary python-dotenv
    ```
    
    ### 🗄️ PostgreSQL Setup
    - Install PostgreSQL locally
    - Create database: `langchain`
    - Update credentials in code if needed
    
    ### 🎥 Supported Videos
    - YouTube videos with **English or Hindi subtitles** (manual or auto-generated)
    - Look for **CC icon** under video player
    
    ### 🚀 Run App
    ```
    streamlit run app.py
    ```
    """)

st.markdown("---")
st.markdown("*Made with ❤️ using yt-dlp for reliable transcript extraction*")
