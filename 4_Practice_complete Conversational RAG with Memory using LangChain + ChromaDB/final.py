import os
import numpy as np
from typing import List
from dotenv import load_dotenv

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Document Loaders
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

# Chains
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables (.env file)
load_dotenv()

# ==========================================================
# LOAD DOCUMENTS
# ==========================================================
def load_documents(path):
    """
    Method Name : load_documents
    Description : Loads a single file or an entire folder.
    Input       : File path or Folder path
    Output      : List of LangChain Documents
    """
    try:
        documents = []
        path = path.strip()

        if not os.path.exists(path):
            raise FileNotFoundError(f"The path does not exist: {path}")

        # --- SINGLE FILE ---
        if os.path.isfile(path):
            extension = os.path.splitext(path)[1].lower()

            if extension == ".txt":
                loader = TextLoader(path, encoding="utf-8")
            elif extension == ".pdf":
                loader = PyPDFLoader(path)
            elif extension == ".docx":
                loader = Docx2txtLoader(path)
            else:
                raise ValueError(f"Unsupported file type: '{extension}'")

            documents.extend(loader.load())

        # --- DIRECTORY ---
        elif os.path.isdir(path):
            # TXT
            documents.extend(
                DirectoryLoader(
                    path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                ).load()
            )
            # PDF
            documents.extend(
                DirectoryLoader(
                    path,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                ).load()
            )
            # DOCX
            documents.extend(
                DirectoryLoader(
                    path,
                    glob="**/*.docx",
                    loader_cls=Docx2txtLoader
                ).load()
            )

        print(f"\nLoaded {len(documents)} document(s)")
        return documents

    except Exception as e:
        print(f"Error Loading Documents : {e}")
        raise

# ============================================================
# STEP 1 : LOAD DOCUMENT
# ============================================================
file_path = input("Enter the document path: ")
documents = load_documents(file_path)
print("\nDocument Loaded Successfully")

# ============================================================
# STEP 2 : CHUNKING
# ============================================================
# FIXED: Removed the explicit single space separator so fallback splitting works intelligently
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print("Chunking completed!")

# ============================================================
# STEP 3 : EMBEDDINGS
# ============================================================
embedding_model = OpenAIEmbeddings()

# ============================================================
# STEP 4 : CHROMADB
# ============================================================
persist_directory = "./chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory,
    collection_name="rag_collection"
)

# ============================================================
# STEP 5 : RETRIEVER
# ============================================================
# FIXED: Changed search_kwarg to search_kwargs
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3} 
)

# ============================================================
# STEP 6 : LLM
# ============================================================
llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# ============================================================
# STEP 7 : HISTORY AWARE RETRIEVER PROMPT
# ============================================================
system_prompt = (
    "Given the chat history and latest user question, rewrite the question so that "
    "it becomes a standalone question. Do not answer the question. Only rewrite it if needed."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# ============================================================
# STEP 8 : HISTORY AWARE RETRIEVER
# ============================================================
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# ============================================================
# STEP 9 : QA PROMPT
# ============================================================
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.\n\n"
            "Use the retrieved context to answer.\n\n"
            "If the answer is not available in the context, simply say: \"I don't know.\"\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# ============================================================
# STEP 10 : DOCUMENT CHAIN
# ============================================================
document_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

# ============================================================
# STEP 11 : FINAL CONVERSATIONAL RAG CHAIN
# ============================================================
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    document_chain
)

# ============================================================
# STEP 12 : MEMORY
# ============================================================
chat_history = []

# ============================================================
# STEP 13 : CHAT LOOP
# ============================================================
print("\nConversational RAG Started")
print("Type 'exit' to stop\n")

while True:
    question = input("\nYou : ")

    if question.lower().strip() == "exit":
        break

    if not question.strip():
        continue

    result = rag_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history
        }
    )

    answer = result["answer"]

    # Display complete conversation turn
    print("\n" + "="*50)
    print(f"User : {question}")
    print(f"Bot  : {answer}")
    print("="*50)

    # Append to memory
    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=answer)
        ]
    )