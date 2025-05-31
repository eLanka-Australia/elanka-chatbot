import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

# Load OpenAI API key from secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Rebuild vector index on the fly
@st.cache_resource
def load_qa_chain():
    loader = TextLoader("elanka_content.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

qa_chain = load_qa_chain()

# Streamlit UI
st.set_page_config(page_title="eLanka Chatbot", page_icon="🧠")
st.title("🧠 Ask anything about eLanka")
st.markdown("This chatbot is powered by content from [eLanka.com.au](https://www.elanka.com.au).")

query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
        st.success("Answer:")
        st.write(result)
