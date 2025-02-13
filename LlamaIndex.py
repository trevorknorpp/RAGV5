import logging
import sys
import os
import asyncio
import faiss
import gradio as gr

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

#FAISS INDEX LOGGING
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# dimensions of mxbai-embed-large-v1
#d = 512
#faiss_index = faiss.IndexFlatL2(d)

repoPath = r"C:\Users\ASLS\Desktop\Trev-LLM\RAGV4\TestFiles"
gitIgnorePath = r"C:\Users\ASLS\source\asls\bioapp\.gitignore"

#Index persistence path
faissIndexPath = "faiss_storage"

# Settings control global defaults
d = 512
Settings.embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

# ----------------------------------
# Index building and loading functions
# ----------------------------------
def build_index() -> VectorStoreIndex:
    """
    Builds the FAISS index from documents and persists it to disk.
    """
    logging.info(f"Building index from documents in: {repoPath}")
    
    # 1) Load documents from your local directory
    documents = SimpleDirectoryReader(repoPath).load_data()
    
    # 2) Instantiate FAISS index with the correct dimension
    faiss_index = faiss.IndexFlatL2(d)
    
    # 3) Create FaissVectorStore by passing faiss_index explicitly
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    # 4) Build a StorageContext from this vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 5) Create the VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    # 6) Persist the index to disk
    index.storage_context.persist(persist_dir=faissIndexPath)
    
    logging.info(f"Index persisted to {faissIndexPath}")
    return index

# -----------------------------------------------------------
# Load the Index
# -----------------------------------------------------------
def load_index() -> VectorStoreIndex:
    """
    Loads a FAISS index from the persisted directory.
    """
    logging.info(f"Loading index from: {faissIndexPath}")
    vector_store = FaissVectorStore.from_persist_dir(persist_dir=faissIndexPath)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    return index

# -----------------------------------------------------------
# Helper Function: get_index()
# -----------------------------------------------------------
def get_index() -> VectorStoreIndex:
    """
    Checks if an index already exists on disk; if yes, loads it;
    otherwise builds a new one.
    """
    if os.path.exists(faissIndexPath) and os.listdir(faissIndexPath):
        return load_index()
    else:
        return build_index()

# ----------------------------------
# Initialize index and query engine
# ----------------------------------
index = get_index()
query_engine = index.as_query_engine(llm=Settings.llm)

async def search_documents(query: str) -> str:
    """Async function to search documents with the query engine."""
    response = await query_engine.aquery(query)
    return str(response)

# ----------------------------------
# Build the agent
# ----------------------------------
agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt=(
        "You are a helpful assistant that can search through a codebase and answer "
        "relevant questions with deep analysis."
    ),
)

# ----------------------------------
# Synchronous wrapper for the agent (for Gradio)
# ----------------------------------
def query_agent(user_query: str) -> str:
    """
    Runs the agent on the given query and returns the response.
    This function wraps the async call in a new event loop.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(agent.run(user_query))
        loop.close()
        return response
    except Exception as e:
        return f"Error: {e}"

# ----------------------------------
# Gradio Interface
# ----------------------------------
iface = gr.Interface(
    fn=query_agent,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here...", label="Query"),
    outputs="text",
    title="Codebase Query Agent",
    description="Ask questions about the codebase or perform calculations.",
)

if __name__ == "__main__":
    index = get_index()
    #iface.launch()