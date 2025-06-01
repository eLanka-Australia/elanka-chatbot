import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import time

# 🔐 Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your real key or use environment vars in PythonAnywhere

# 📍 Your sitemap
SITEMAP_URL = "https://www.elanka.com.au/sitemap.xml"

# 🧭 Step 1: Fetch all URLs from the sitemap
def get_sitemap_urls(sitemap_url):
    response = requests.get(sitemap_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch sitemap")
    root = ET.fromstring(response.content)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [elem.text for elem in root.findall(".//ns:loc", ns)]

# 🧹 Step 2: Scrape content from each page
def extract_content_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return f"URL: {url}\n{text}"
    except Exception as e:
        return f"URL: {url}\n[ERROR fetching content: {e}]"

# 💾 Step 3: Save combined content
def save_content_to_file(contents, filename="elanka_content.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in contents:
            f.write(entry + "\n\n")

# 🤖 Step 4: Rebuild vector index
def rebuild_faiss_index(text_file="elanka_content.txt", output_folder="elanka_full_index"):
    loader = TextLoader(text_file, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(output_folder)
    print(f"✅ Rebuilt vector index with {len(docs)} chunks.")

# 🔁 Orchestrate the full pipeline
def main():
    print("📡 Fetching sitemap...")
    urls = get_sitemap_urls(SITEMAP_URL)
    print(f"✅ Found {len(urls)} URLs.")

    print("🧹 Extracting content...")
    contents = [extract_content_from_url(url) for url in urls]
    save_content_to_file(contents)

    print("🧠 Rebuilding vector index...")
    rebuild_faiss_index()

    print("🎉 Done. Content and index refreshed!")

if __name__ == "__main__":
    main()
