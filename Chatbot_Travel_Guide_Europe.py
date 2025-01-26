#from My_HF_TOKEN import HF_TOKEN
import os
# Access the secret as an environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

import streamlit as st

# llm
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model, huggingfacehub_api_token=HF_TOKEN)

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/contents/"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
vector_db_combined = FAISS.load_local("contents/faiss_index_europeLens", embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vector_db_combined.as_retriever(search_kwargs={"k": 2})

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


