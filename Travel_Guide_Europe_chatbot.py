# %% [markdown]
# # Rag chatbot
# Let's take the next step in our journey! We've explored LLMs, chatbots, and RAG. Now, it's time to put them all together to create a powerful tool: a RAG chain with memory.

# %% [markdown]
# ---
# ## 1.&nbsp; Installations and Settings 🛠️
# LangChain is a framework that simplifies the development of applications powered by large language models (LLMs). Here we install their HuggingFace package as we'll be using open source models from HuggingFace.

# %%
pip install -qqq -U langchain-huggingface
pip install -qqq -U langchain
pip install -qqq -U langchain-community
pip install -qqq -U faiss-cpu

# %% [markdown]
# To use the LLMs, you'll need to create a HuggingFace access token for this project.
# 1. Sign up for an account at [HuggingFace](https://huggingface.co/)
# 2. Go in to your account and click `edit profile`
# 3. Go to `Access Tokens` and create a `New Token`
# 4. The `Type` of the new token should be set to `Read`
# 
# We've then saved ours as a Colab secret - this way we can use it in multiple notebooks without having to type it or reveal it.

# %%
#From Colab:
#import os
#from google.colab import userdata # we stored our access token as a colab secret

# Set the token as an environ variable
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = userdata.get('HF_TOKEN')

# %%
# In VSCode:
# Define the HF_TOKEN to be passed into the password in a .py file called HF_TOKEN.
from My_HF_TOKEN import HF_TOKEN

# %% [markdown]
# ---
# ## 2.&nbsp; Setting up your LLM 🧠

# %%
from langchain_huggingface import HuggingFaceEndpoint

# This info's at the top of each HuggingFace model page
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(repo_id = hf_model, huggingfacehub_api_token=HF_TOKEN)

# %% [markdown]
# ---
# ## 3.&nbsp; Retrieval Augmented Generation 🔃

# %% [markdown]
# ### 3.1.&nbsp; Find your data
# Our model needs some information to work its magic! In this case, we'll be using Spain travel information in pdf format

# %%
file_path1 = (
    "Espana_General.pdf"
)

# %%
file_path2 = (
    "Ultimate-Travel-Guide-Europe.pdf"
)

# %% [markdown]
# ### 3.2.&nbsp; Load the data

# %%
%pip install -qU pypdf

# %%
#Load the data
from langchain_community.document_loaders import PyPDFLoader

# List of file paths
file_paths = [file_path1, file_path2]

# Combined list of Document objects
combined_pages = []

# Asynchronous function to load pages from multiple PDFs
async def load_pages_from_pdfs(file_paths):
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        async for page in loader.alazy_load():
            combined_pages.append(page)

# Call the function with the list of file paths
await load_pages_from_pdfs(file_paths)

# Now `combined_pages` contains Document objects from both PDFs

# %%
print(f"{combined_pages[1].metadata}\n")
print(combined_pages[1].page_content)

# %%
print(f"{combined_pages[60].metadata}\n")
print(combined_pages[60].page_content)

# %% [markdown]
# ### 3.3.&nbsp; Splitting the document

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                               chunk_overlap=150)

docs_combined = text_splitter.split_documents(combined_pages)

# %% [markdown]
# ### 3.4.&nbsp; Creating vectors with embeddings

# %%
from langchain_huggingface import HuggingFaceEmbeddings

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/content/"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

# %%
#From google Colab
#from google.colab import drive
#drive.mount('/content/drive')

# %% [markdown]
# ### 3.5.&nbsp; Creating a vector database

# %%
from langchain.vectorstores import FAISS

vector_db_combined = FAISS.from_documents(docs_combined, embeddings)

# %%
vector_db_combined.save_local("/content/faiss_index_combined")

# %%
#From google Colab
#vector_db("/content/drive/MyDrive/GenAI-Faiss/vector_db", index=False) #save to your drive /My Drive

# %%
vector_db_combined.similarity_search("What is the most delicious food in Spain?")

# %%
vector_db_combined.similarity_search("Name 3 of the most famous country in Europe?")

# %% [markdown]
# ---
# ## 4.&nbsp; Setting up the chain 🔗
# There are three cooperating pieces here to work with:
# - `create_history_aware_retriever` chains together an llm, retriever, and prompt. It is similar to `RetrievalQA`.
# - `create_stuff_documents_chain` again calls on your llm and prompt to respond conversationally to you in a retrieval context.
# - `create_retrieval_chain` chains the other two pieces together.
# 
# Notice that this time, `chat_history` is allowed to be a simple list, rather than a class with methods to manipulate the history.
# 
# Finally, when invoking our RAG chatbot, it is important for the call to reference the chat history explicitly.

# %%
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

#Setting up your LLM:
# hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
# llm = HuggingFaceEndpoint(repo_id=hf_model)

#Creating vectors with embeddings:
# embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
# embeddings_folder = "/content/"

# embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
#                                   cache_folder=embeddings_folder)

vector_db = FAISS.load_local("/content/faiss_index_combined", embeddings, allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever(search_kwargs={"k": 2}) # top 2 results only, speed things up

#Adding a prompt:
#We can guide our model's behavior with a prompt, similar to how we gave instructions to the chatbot.
template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {input}
Response:"""

chat_history = []

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

doc_retriever = create_history_aware_retriever(
    llm, retriever, prompt
)

doc_chain = create_stuff_documents_chain(llm, prompt)

rag_bot = create_retrieval_chain(
    doc_retriever, doc_chain
)

# %%
ans = rag_bot.invoke({"input":"What can I eat in Spain?", "chat_history": chat_history, "context": retriever})
chat_history.extend([{"role": "human", "content": ans["input"]},{"role": "assistant", "content":ans["answer"]}])

# %%
print(ans['answer'])

# %%
ans = rag_bot.invoke({"input": "Name 3 of the most famous country in Europe?", "chat_history": chat_history, "context": retriever})
chat_history.extend([{"role": "human", "content": ans["input"]},{"role": "assistant", "content":ans["answer"]}])
print(ans["answer"])

# %% [markdown]
# We can also use a while loop here to make our chatbot a little more interactive.

# %%
chain_2 = create_retrieval_chain(
    doc_retriever, doc_chain
)
history = []
# Start the conversation loop
while True:
  user_input = input("You: ")

  # Check for exit condition
  if user_input.lower() == 'end':
      print("Ending the conversation. Goodbye!")
      break

  # Get the response from the conversation chain
  response = chain_2.invoke({"input":user_input, "chat_history": history, "context": retriever})
  history.extend([{"role": "human", "content": response["input"]},{"role": "assistant", "content":response["answer"]}])
  # Print the chatbot's response
  print(response["answer"])

# %% [markdown]
# ---
# ## 5.&nbsp; Streamlit
# Streamlit lets you transform your data scripts into interactive dashboards and prototypes in minutes, without needing front-end coding knowledge.

# %% [markdown]
# We first need to install [streamlit](https://streamlit.io/) - as always, locally this is a one time thing, whereas on colab we need to do it each session.

# %%
#%pip install -q streamlit

# %% [markdown]
# To run Streamlit on Colab, we'll have to set up a tunnel. If you're working locally, you can skip this step.     
# Modified from [Source](https://colab.research.google.com/gist/thapecroth/67a69d840010ffcfe7523655808c5b92/streamlit-on-colab.ipynb).

# %%
# code necessary for Colab only

# import os
# import time
# from IPython.display import display, HTML
# def tunnel_prep():
#     for f in ('cloudflared-linux-amd64', 'logs.txt', 'nohup.out'):
#         try:
#             os.remove(f'/content/{f}')
#             print(f"Deleted {f}")
#         except FileNotFoundError:
#             continue

#     !wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -q
#     !chmod +x cloudflared-linux-amd64
#     !nohup /content/cloudflared-linux-amd64 tunnel --url http://localhost:8501 &
#     url = ""
#     while not url:
#         time.sleep(1)
#         result = !grep -o 'https://.*\.trycloudflare.com' nohup.out | head -n 1
#         if result:
#             url = result[0]
#     return display(HTML(f'Your tunnel URL <a href="{url}" target="_blank">{url}</a>'))

# %% [markdown]
# Here's an example of what you can produce with streamlit. It's so easy, just a few lines of python depending on what you want, and so many options!
# 
# - Locally you would write this script in a .py file and not a notebook (.ipynb).
# 
# - On colab, we can create a .py file by using the magic command `%%writefile` at the top of the cell. This command writes the cell content to a file, naming it 'app.py', or whatever else you choose, in this instance. Once saved, you can see 'app.py' in Colab's storage by clicking on the left-hand side folder icon.

# %% [markdown]
# ---
# ## 6.&nbsp; RAG chatbot in streamlit ⭐️

# %% [markdown]
# Now, let's proceed by creating the .py file for our rag chatbot.
# 
# We sourced the foundational code for our Streamlit basic chatbot from the [Streamlit documentation](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps).
# 
# In addition, we implemented [cache_resource](https://docs.streamlit.io/library/api-reference/performance/st.cache_resource) for both memory and LLM. Given that Streamlit reruns the entire script with each message input, relying solely on memory would result in data overwriting and a loss of conversational continuity. The inclusion of cache resource prevents Streamlit from creating a new memory instance on each run. This was also added to the LLM, enhancing speed and preventing its reload in every iteration.

# %%
#From google Colab:  %%writefile rag_app.py

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import streamlit as st

# llm
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/content/"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
vector_db_combined = FAISS.load_local("/content/faiss_index_combined", embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vector_db_combined.as_retriever(search_kwargs={"k": 2})

# prompt
template = """You are a professional chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

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

# %% [markdown]
# We wrote the code in a .py file in the same folder and run the following code:

# %%
#Local:
%run RAG_Chatbot_combined.py

# %% [markdown]
# Now, let's see what we've made.

# %% [markdown]
# ### Local

# %% [markdown]
# Run this code in a terminal, following the caveats as laid out previously.
# ```
# streamlit run rag_app.py
# ```

# %%
#Run this code in a terminal: 
#Note: First check the address in terminal
# streamlit run RAG_Chatbot_combined.py

# %% [markdown]
# ### Colab

# %%
#tunnel_prep()

# %%
#!streamlit run rag_app.py &>/content/logs.txt &


