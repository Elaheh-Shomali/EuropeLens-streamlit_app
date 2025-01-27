from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import streamlit as st

import os
# Access the secret as an environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# llm
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model, huggingfacehub_api_token = HF_TOKEN)

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "content/"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
vector_db = FAISS.load_local("content/faiss_index_combined", embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# prompt
template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {input}
Response:"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# bot with memory
@st.cache_resource
def init_bot():
    doc_retriever = create_history_aware_retriever(llm, retriever, prompt)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(doc_retriever, doc_chain)

rag_bot = init_bot()


##### streamlit #####

st.title("EuropeLens")

# Markdown
st.markdown("""
Welcome to EuropeLens! 🌟

Ask me anything about Europe, and I’ll bring you the answers—culture, history, travel tips, and more!

""")

# Map
map_data = [
    {"name": "Paris", "lat": 48.856613, "lon": 2.352222},
    {"name": "London", "lat": 51.507351, "lon": -0.127758},
    {"name": "Rome", "lat": 41.902783, "lon": 12.496366},
    {"name": "Barcelona", "lat": 41.385064, "lon": 2.173404},
    {"name": "Amsterdam", "lat": 52.367573, "lon": 4.904139},
    {"name": "Berlin", "lat": 52.520008, "lon": 13.404954},
    {"name": "Prague", "lat": 50.075539, "lon": 14.437800},
    {"name": "Vienna", "lat": 48.210033, "lon": 16.363449},
    {"name": "Madrid", "lat": 40.416775, "lon": -3.703790},
    {"name": "Lisbon", "lat": 38.716667, "lon": -9.139089},
    {"name": "Venice", "lat": 45.440847, "lon": 12.315515},
    {"name": "Florence", "lat": 43.769562, "lon": 11.255814},
    {"name": "Istanbul", "lat": 41.008240, "lon": 28.978359},
    {"name": "Stockholm", "lat": 59.329323, "lon": 18.068581},
    {"name": "Copenhagen", "lat": 55.676098, "lon": 12.568337},
    {"name": "Edinburgh", "lat": 55.953251, "lon": -3.188267},
    {"name": "Dublin", "lat": 53.349805, "lon": -6.260310},
    {"name": "Brussels", "lat": 50.850346, "lon": 4.351721},
    {"name": "Budapest", "lat": 47.497913, "lon": 19.040236},
    {"name": "Zurich", "lat": 47.376887, "lon": 8.541694},
    {"name": "Munich", "lat": 48.135125, "lon": 11.581981},
    {"name": "Athens", "lat": 37.983810, "lon": 23.727539},
    {"name": "Milan", "lat": 45.464203, "lon": 9.189982},
    {"name": "Nice", "lat": 43.710173, "lon": 7.261953},
    {"name": "Seville", "lat": 37.388630, "lon": -5.982320},
    {"name": "Porto", "lat": 41.157944, "lon": -8.629105},
    {"name": "Helsinki", "lat": 60.169856, "lon": 24.938379},
    {"name": "Warsaw", "lat": 52.229676, "lon": 21.012229},
    {"name": "Oslo", "lat": 59.913868, "lon": 10.752245},
    {"name": "Reykjavik", "lat": 64.135484, "lon": -21.895411},
    {"name": "Krakow", "lat": 50.064651, "lon": 19.944981},
    {"name": "Dubrovnik", "lat": 42.650660, "lon": 18.094420},
    {"name": "Saint Petersburg", "lat": 59.934280, "lon": 30.335099},
]

st.map(map_data)

# Initialise chat history
# Chat history saves the previous messages to be displayed
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Curious minds wanted!"):

    # Display user message in chat message container
    st.chat_message("human").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "human", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Diving deep to uncover the truth..."):

        # send question to chain to get answer
        answer = rag_bot.invoke({"input": prompt, "chat_history": st.session_state.messages, "context": retriever})

        # extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content":  response})
