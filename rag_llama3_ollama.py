from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import faiss
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
import numpy as np
import os

# 🔹 1. Load PDF and split into chunks
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# 🔹 2. Create Embeddings and FAISS Vector DB
def create_vectorstore(documents):
    # 1. Prepare texts
    texts = [doc.page_content for doc in documents]

    # 2. Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


    # 3. Embed documents
    embeddings = embedding_model.embed_documents(texts)
    embeddings_np = np.array(embeddings).astype("float32")

    # 4. FAISS IndexIVFFlat setup
    dimension = embeddings_np.shape[1]
    nlist = min(10, len(embeddings_np))  # avoid 'nx >= k' error
    quantizer = faiss.IndexFlatL2(dimension)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    # 5. Train + Add embeddings
    if not index_ivf.is_trained:
        index_ivf.train(embeddings_np)
    index_ivf.add(embeddings_np)

    # 6. Setup LangChain FAISS interface
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    for i, doc in enumerate(documents):
        doc_id = str(i)
        docstore.add({doc_id: doc})
        index_to_docstore_id[i] = doc_id

    # 7. Return LangChain FAISS object
    vectorstore = FAISS(
        index=index_ivf,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_model
    )

    return vectorstore

# 🔹 3. Create RAG Chain with Ollama (LLaMA 3)
def create_rag_chain(vectorstore):
    llm = ChatOllama(
        model="llama3",  
        temperature=0.7
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# 🔹 4. ask_question
def ask_question(chain, query):
    result = chain(query)
    print("\n🧠 Answer:\n", result["result"])
    print("\n📚 Prompt:")
    for doc in result["source_documents"]:
        print("—", doc.metadata["source"])

# 🔹 5. Run everything
if __name__ == "__main__":
    pdf_path = "Depression.pdf"  
    print("🔍 load_and_split_pdf...")
    split_docs = load_and_split_pdf(pdf_path)

    print("📦 create_vectorstore...")
    vectorstore = create_vectorstore(split_docs)

    print("🤖 LLaMA 3 (Ollama)...")
    rag_chain = create_rag_chain(vectorstore)

    while True:
        query = input("\n❓ Ask a question (or type 'exit' to quit):")
        if query.lower() == "exit":
            break
        ask_question(rag_chain, query)
