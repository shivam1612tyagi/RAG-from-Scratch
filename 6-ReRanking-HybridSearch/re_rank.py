import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# STEP 1 & 2: LOADING & CHUNKING
# ============================================================================
# 1. Load the single source text file directly using TextLoader
loader = TextLoader(file_path="langchain_sample.txt", encoding="utf-8")
documents = loader.load()

# 2. Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"✅ Loaded and split into {len(chunks)} chunks.")

# ============================================================================
# STEP 3 & 4: DENSE RETRIEVAL LAYER (ChromaDB Vector Store)
# ============================================================================
embedding_model = OpenAIEmbeddings()

# Built with correct argument names and valid collection_name formatting
vectorstore = Chroma.from_documents(
    documents=chunks,              
    embedding=embedding_model,
    collection_name="rag_collection",
    persist_directory="./chroma_db_reranking"
)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ============================================================================
# STEP 5 & 6: SPARSE & HYBRID RETRIEVAL SETUP
# ============================================================================
# Sparse keyword retriever index
sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 5

# Merge both retrievers seamlessly using EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)
print("✅ Hybrid retrieval system built successfully.")

# ============================================================================
# STEP 7 & 8: EXECUTE INITIAL RETRIEVAL & LLM RERANKING
# ============================================================================
query = "How can i use langchain to build an application with memory and tools?"
print(f"\n🔍 Searching for: '{query}'")

# 1. Fetch initial broad list of document candidates
retrieved_docs = hybrid_retriever.invoke(query)

# 2. Format them clearly with 1-based numeric bullet points for the LLM to read
doc_lines = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
formatted_docs = "\n".join(doc_lines)

# 3. Prompt the LLM strictly to prioritize and sort indices
rerank_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Your task is to rank the following documents from most to least relevant to the user's question.

User Question: "{question}"

Documents:
{documents}

Instructions:
- Think about the relevance of each document to the user's question.
- Return a list of document indices in ranked order, starting from the most relevant.

Output format: comma-separated document indices (e.g., 2,1,3,0,...)
""")

llm = init_chat_model("openai:gpt-3.5-turbo")
rerank_chain = rerank_prompt | llm | StrOutputParser()

# 4. Get the ordering string back from the LLM (e.g., "3,1,5,4,2")
ranking_response = rerank_chain.invoke({
    "question": query,
    "documents": formatted_docs
})
print(f"🤖 LLM Reranker Order recommendation: {ranking_response}")

# ============================================================================
# STEP 8.5: PARSE LLM STRING AND ASSIGN CODES TO PARTICULAR POSITIONS
# ============================================================================
reranked_docs = []
try:
    # Convert string indices (1-indexed from prompt) down into 0-indexed integers
    rank_indices = [int(x.strip()) - 1 for x in ranking_response.split(",") if x.strip().isdigit()]
    
    # Reassign each particular document to its recommended slot
    for idx in rank_indices:
        if 0 <= idx < len(retrieved_docs):
            reranked_docs.append(retrieved_docs[idx])
            
    # Guard-rail: If the LLM left out any candidates, safely append them to the end
    for doc in retrieved_docs:
        if doc not in reranked_docs:
            reranked_docs.append(doc)
            
    print(f"📊 Successfully reordered {len(reranked_docs)} documents into targeted positions.")
except Exception as e:
    print(f"⚠️ Parsing failed, falling back to original retriever order: {e}")
    reranked_docs = retrieved_docs

# ============================================================================
# STEP 9: FINAL ANSWER GENERATION (THE "RAG" SYNTHESIS)
# ============================================================================
generation_prompt = PromptTemplate.from_template("""
You are an expert AI developer assistant. Answer the user's question using ONLY the provided context below. 
If the context doesn't contain the answer, politely state that you don't know.

Context:
{context}

User Question: {question}

Final Answer:
""")

generation_chain = generation_prompt | llm | StrOutputParser()

# Pass only the top 3 highest-rated documents to feed our final response context
top_n_context = "\n\n".join([doc.page_content for doc in reranked_docs[:3]])

final_answer = generation_chain.invoke({
    "question": query,
    "context": top_n_context
})

print("\n🚀 Final Synthesized Answer:")
print(final_answer)