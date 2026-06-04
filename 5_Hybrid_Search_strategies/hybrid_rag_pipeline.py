import os
import sys

# ============================================================================
# ENVIRONMENT SETUP & PATH RESOLUTION
# ============================================================================
# Forces Python to look into the local virtual environment's site-packages first.
# This prevents IDE/Jupyter kernel path mismatch issues with installed libraries.
venv_path = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# Verify core dependencies before starting execution
try:
    import faiss
    from langchain_community.retrievers import BM25Retriever
    print("🎉 Success! Core retrieval dependencies (FAISS, BM25) are verified.")
except ImportError as e:
    print(f"⚠️ Critical Dependency Missing: {e}")
    sys.exit(1)

# Remaining imports
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import init_chat_model
from langchain_classic.chains import create_retrieval_chain

# Document Loaders
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

# ============================================================================
# STEP 1: DOCUMENT LOADING
# ============================================================================
# Load raw text documents from the temporary directory path
data_directory = r"C:\Users\shiva\AppData\Local\Temp\tmp8qb1tq5f"

print(f"\nScanning and loading text files from: {data_directory}...")
loader = DirectoryLoader(
    path=data_directory,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

documents = loader.load()
print(f"Successfully loaded {len(documents)} source document(s).")

# ============================================================================
# STEP 2: TEXT SPLITTING & CHUNKING
# ============================================================================
# RecursiveCharacterTextSplitter cleanly splits large files down into manageable sizes
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} structural chunks from raw documents.")

# ============================================================================
# STEP 3: EMBEDDINGS ENGINE INITIALIZATION
# ============================================================================
# Uses OpenAI's text-embedding-ada-002 model by default to convert text into math vectors
# (Ensure your OPENAI_API_KEY environment variable is set before running)
embedding_model = OpenAIEmbeddings()

# ============================================================================
# STEP 4: VECTOR STORE SETUP (DENSE RETRIEVAL LAYER)
# ============================================================================
# Initialize and build a local directory-persisted Chroma vector store index
persist_dir = "./chroma_db"
print(f"Indexing chunks inside ChromaDB vector store at: {persist_dir}...")

vectorstore = Chroma.from_documents(
    documents=chunks,  # Passing split chunks for high semantic alignment
    embedding=embedding_model,
    collection_name="RAG-Collection",
    persist_directory=persist_dir
)

# Convert vectorstore into a basic Dense semantic search retriever
dense_retriever = vectorstore.as_retriever()

# ============================================================================
# STEP 5: BM25 SEARCH SETUP (SPARSE RETRIEVAL LAYER)
# ============================================================================
# Initialize keyword matcher using BM25 over the exact same chunk slices
print("Initializing Sparse BM25 Keyword Search Index...")
sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 3  # Retrieve the top 3 keyword-matched document layers

# ============================================================================
# STEP 6: ENSEMBLE HYBRID ENGINE COMPILATION
# ============================================================================
# Melds semantic context lookup (Chroma) and precise keywords (BM25) using RRF
# Weights give 70% importance to conceptual context, 30% to word matching
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)
print("Hybrid Ensemble search algorithm bound successfully.")

# ============================================================================
# STEP 7: PROMPT & LLM DEFINITIONS
# ============================================================================
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("human", "{input}")
])

# Initialize the chat model using the unified standard factory pattern
llm = init_chat_model("openai:gpt-3.5-turbo")

# Create the standard multi-document synthesis formatting chain layer
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# ============================================================================
# STEP 8: FINAL LCEL RAG CHAIN COMPILATION
# ============================================================================
# Ties the hybrid document finder to the synthesis LLM engine layout
rag_chain = create_retrieval_chain(
    retriever=hybrid_retriever, 
    combine_docs_chain=document_chain
)

# ============================================================================
# STEP 9: EXECUTION & TEST QUERY RUNNER
# ============================================================================
query_payload = {"input": "How can I build an app using LLMs?"}
print(f"\nInvoking Hybrid RAG Pipeline for query: '{query_payload['input']}'...\n")

response = rag_chain.invoke(query_payload)

# Display synthesized output answer
print("=" * 60)
print("✅ Answer:")
print(response["answer"])
print("=" * 60)

# Display source tracing reference documents used by the chain
print("\n📄 Source Documents Utilized:")
for i, doc in enumerate(response["context"]):
    # Pull file identifier cleanly from absolute metadata paths
    source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
    print(f"\n[Doc {i+1}] Source: {source_name}")
    print(f"Content snippet: {doc.page_content.strip()[:200]}...")