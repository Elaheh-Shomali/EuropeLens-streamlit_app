from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import streamlit as st

import os
# Access the secret as an environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# llm
hf_model = "Qwen/Qwen2.5-7B-Instruct-1M"
llm = HuggingFaceEndpoint(repo_id=hf_model, 
                          huggingfacehub_api_token = HF_TOKEN #, 
                          #task="chat"  # explicitly specify the task
                         )

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

Ask me anything about Europe, and I'll bring you the answers—culture, attractions, travel tips, and more!

""")

# Adding an image to the main page
st.image(
    "Photo/europa.png",
    use_container_width=True
)

# CSS to change the background color: #c1f0c1 #f5f5f5 #e0e0e0 #d6d6d6
# Apply custom CSS for colored expanders

st.markdown("""
    <style>
    /* Set the background color for the entire app */
    body {
        background-color: #E0D0FF;  /* Replace with your desired color */
        color: #333333;  /* Set default text color */
    }

    /* Customize the Streamlit header */
    .css-18e3th9 {
        font-size: 36px;
        color: #2a2a2a;  /* Set a dark color for the header text */
    }

    /* Customize buttons */
    .stButton button {
        background-color: #ffa500;  /* Set button background color */
        color: white;  /* Set button text color */
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }

    /* Customize the expander header */
    .streamlit-expanderHeader {
        background-color: #ffa500;
        color: white;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 5px;
    }

    </style>
    """, unsafe_allow_html=True)

# City Information with Expanders

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("Madrid"):
        st.write("""
        - **Population**: ~3.3 million
        - **Highlights**: Prado Museum, Royal Palace, and Retiro Park
        - **Known For**: Being Spain's capital, its rich art scene, and lively nightlife.
        """)

with col2:
    with st.expander("Rome"):
        st.write("""
        - **Population**: ~2.8 million
        - **Highlights**: Colosseum, Vatican City, Pantheon, and Roman Forum
        - **Known For**: Ancient history, Vatican City, and iconic landmarks like the Colosseum and St. Peter's Basilica.
        """)

with col3:
    with st.expander("Paris"):
        st.write("""
        - **Population**: ~2.1 million
        - **Highlights**: Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Montmartre
        - **Known For**: The city of romance, world-class art, haute couture fashion, and culinary delights.
        """)

# Second row of columns
col4, col5, col6 = st.columns(3)

with col4:
    with st.expander("London"):
        st.write("""
        - **Population**: ~8.9 million
        - **Highlights**: Buckingham Palace, Tower of London, British Museum, and the London Eye
        - **Known For**: Its history, iconic landmarks like Big Ben, and being a global hub for culture and finance.
        """)

with col5:
    with st.expander("Berlin"):
        st.write("""
        - **Population**: ~3.7 million
        - **Highlights**: Brandenburg Gate, Berlin Wall, Museum Island, and Alexanderplatz
        - **Known For**: A history shaped by division, vibrant art scene, and an iconic nightlife culture.
        """)

with col6:
    with st.expander("Lisbon"):
        st.write("""
        - **Population**: ~0.5 million
        - **Highlights**: Belém Tower, Jerónimos Monastery, Alfama district, and São Jorge Castle
        - **Known For**: Stunning views, historic trams, and vibrant Fado music.
        """)
      
# Map
map_data = [
    {"name": "Paris", "lat": 48.856613, "lon": 2.352222},
    {"name": "London", "lat": 51.507351, "lon": -0.127758},
    {"name": "Rome", "lat": 41.902783, "lon": 12.496366},
    {"name": "Amsterdam", "lat": 52.367573, "lon": 4.904139},
    {"name": "Berlin", "lat": 52.520008, "lon": 13.404954},
    {"name": "Prague", "lat": 50.075539, "lon": 14.437800},
    {"name": "Vienna", "lat": 48.210033, "lon": 16.363449},
    {"name": "Madrid", "lat": 40.416775, "lon": -3.703790},
    {"name": "Lisbon", "lat": 38.716667, "lon": -9.139089},
    {"name": "Stockholm", "lat": 59.329323, "lon": 18.068581},
    {"name": "Copenhagen", "lat": 55.676098, "lon": 12.568337},
    {"name": "Dublin", "lat": 53.349805, "lon": -6.260310},
    {"name": "Brussels", "lat": 50.850346, "lon": 4.351721},
    {"name": "Budapest", "lat": 47.497913, "lon": 19.040236},
    {"name": "Istanbul", "lat": 41.008240, "lon": 28.978359},
    {"name": "Athens", "lat": 37.983810, "lon": 23.727539},
    {"name": "Helsinki", "lat": 60.169856, "lon": 24.938379},
    {"name": "Warsaw", "lat": 52.229676, "lon": 21.012229},
    {"name": "Oslo", "lat": 59.913868, "lon": 10.752245},
]

st.map(map_data)

# Initialize session state variables if they don't exist
if 'restart' not in st.session_state:
    st.session_state.restart = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns(2)

# Button to end the conversation
with col1:
    if st.button("End Conversation"):
        st.session_state.messages.clear()
        st.write("Ending the conversation. Goodbye!")

# Button to restart the conversation
with col2:
    if st.button("Restart Conversation"):
        # Set the flag to restart the conversation
        st.session_state.restart = True

# If restart flag is True, reset the messages and update state
if st.session_state.restart:
    st.session_state.messages = []  # Clear conversation history
    st.session_state.restart = False  # Reset restart flag to prevent continuous reruns
    st.write("Conversation restarted. How can I help you today?")

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
