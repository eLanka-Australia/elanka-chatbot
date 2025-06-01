import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ✅ Pull from GitHub Environment Secret
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# 📍 Your sitemap
SITEMAP_URL = "https://www.elanka.com.au/sitemap.xml"

# 🧭 Step 1: Fetch all URLs from the sitemap
def get_sitemap_urls(sitemap_url):
    response = requests.get(sitemap_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch sitemap")
    soup = BeautifulSoup(response.content, "xml")
    urls = [loc.text for loc in soup.find_all("loc")]
    return urls[:500]  # limit to 500 for speed

# 📄 Step 2: Download HTML + extract text
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator="\n")
    except:
        return ""

# 🧠 Step 3: Build vector index
def rebuild_faiss_index():
    print("📡 Fetching sitemap...")
    urls = get_sitemap_urls(SITEMAP_URL)
    print(f"✅ Found {len(urls)} URLs.")

    print("🧹 Extracting content...")
    documents = []
    for url in urls:
        content = extract_text_from_url(url)
        if content.strip():
            documents.append({"text": content, "metadata": {"source": url}})

    print("🧠 Rebuilding vector index...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([doc["text"] for doc in documents],
                                     metadatas=[doc["metadata"] for doc in documents])

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("elanka_full_index")

    with open("elanka_content.txt", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(f"{doc['metadata']['source']}\n{doc['text']}\n\n")

    print(f"✅ Done. Indexed {len(docs)} chunks.")

# 🚀 Run
def main():
    rebuild_faiss_index()

if __name__ == "__main__":
    main()
