import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load the vector index
@st.cache_resource
def load_qa_chain():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("elanka_full_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

qa_chain = load_qa_chain()

st.set_page_config(page_title="eLanka Chatbot", page_icon="🧠")
st.title("🧠 Ask anything about eLanka")
st.markdown("This chatbot is powered by eLanka.com.au")

query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
        st.success("Answer:")
        st.write(result)
