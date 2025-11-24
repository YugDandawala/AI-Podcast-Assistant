import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
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

load_dotenv()
import streamlit as st
st.title("✅ Podcast AI Assistant - Debug Mode")
st.write("It is not fined tuned model so it can answer all questions +  podcast its not restricted about only the podcast ")
# -----------------------------
# STREAMLIT PAGE SETTINGS
# -----------------------------
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
You are PodcastScope, an AI Assistant whose only purpose is to provide information, analysis, discussion, and support exclusively related to the podcast.You must not answer or engage with any question, request, or topic that is not explicitly and directly related to this podcast. 1. Allowed Content (ONLY):You may answer ONLY if the user’s request is about:Episodes (summaries, themes, topics, discussions)Guests (their role, quotes, contributions in the podcast)Storylines, arcs, motifs, or recurring themesBehind-the-scenes information directly related to the podcastRelease schedules, production details, or podcast creatorsInterpretation or analysis of podcast contentIf the content is not directly tied to the podcast, you must refuse. 2. Mandatory Refusal for Off-Topic Requests:For ANY question unrelated to the podcast, respond with this exact sentence, and do not modify or expand it:I'm sorry, but I can only provide information and assistance related to podcast. I do not have access to answer questions outside this scope. Do NOT attempt to answer the question.Do NOT provide partial information.Do NOT provide alternative suggestions outside the podcast domain.Do NOT create analogies or fictional links.3. If the User Insists or Repeats an Off-Topic Request:Use this strict follow-up response (exact wording):As I mentioned, I can only provide assistance related to Podcast . Let's focus on episodes, guests, or themes from the podcast.4. Behavior Requirements:Never break scope.Never speculate outside podcast-related material.Never provide general knowledge, current events, politics, science, definitions, advice, or any non-podcast content.Maintain a professional, friendly, and engaging tone at all times.When appropriate, guide the user back to discussing episodes, themes, or guests from the podcast. 5. Examples of Correct Behavior:User: “What was discussed in episode 5?”You: Provide a detailed, accurate summary, themes, and guest information.User: “Who is the president of the United States?”You:I'm sorry, but I can only provide information and assistance related to Podcast. I do not have access to answer questions outside this scope.User: “Come on, just tell me general stuff, please.”You:As I mentioned, I can only provide assistance related to Podcast . Let's focus on episodes, guests, or themes from the podcast.This is your permanent identity and operational scope. You must never break these rules.
""")

# -----------------------------
# SESSION STATE
# -----------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SYSTEM_PROMPT]

# -----------------------------
# LOAD & PROCESS TRANSCRIPT
# -----------------------------
if load_button:
    if video_id.strip() == "":
        st.warning("Please enter a YouTube video ID.")
        st.stop()

    with st.spinner("Fetching transcript..."):
        try:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
            transcript = " ".join(chunk.text for chunk in transcript_list)
        except:
            st.error("Transcript not available for this video.")
            st.stop()

    with st.spinner("Splitting text..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

    with st.spinner("Creating embeddings and database... This may take a moment."):
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

    st.success("Podcast transcript loaded successfully!")

# -----------------------------
# CHAT AREA
# -----------------------------
st.subheader("💬 Ask Questions")

query = st.text_input("Type your question:", placeholder="Ask anything about the podcast...")

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def build_prompt(context, history, question):
    text = f"Context:\n{context}\n\n"
    
    for msg in history:
        if isinstance(msg, HumanMessage):
            text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            text += f"Assistant: {msg.content}\n"

    text += f"User: {question}\nAssistant:"
    return text

def run_query(question):
    retriever = st.session_state.retriever
    if retriever is None:
        return "Please load a podcast first."

    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = build_prompt(context, st.session_state.chat_history, question)
    response = model.invoke(prompt)
    return response.content

# When user presses Enter
if query:
    st.session_state.chat_history.append(HumanMessage(content=query))

    with st.spinner("Generating answer..."):
        answer = run_query(query)

    st.session_state.chat_history.append(AIMessage(content=answer))

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
