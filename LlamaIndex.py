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
d = 512
faiss_index = faiss.IndexFlatL2(d)

repoPath = r"C:\Users\ASLS\Desktop\Trev-LLM\RAGV4\TestFiles"
gitIgnorePath = r"C:\Users\ASLS\source\asls\bioapp\.gitignore"

#Index persistence path
faissIndexPath = "faiss_storage"

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)


#----------------------------------
# Injest and Index the Entire Repo
#----------------------------------
documents = SimpleDirectoryReader(repoPath).load_data()

# Create FAISS vector store
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
    #more repos can be added if needed
    #we can also override the embedding model here
    )

# save index to disk
#index.storage_context.persist()

# load index from disk
#vector_store = FaissVectorStore.from_persist_dir("./storage")
#storage_context = StorageContext.from_defaults(
#    vector_store=vector_store, persist_dir="./storage"
#)
#index = load_index_from_storage(storage_context=storage_context)


query_engine = index.as_query_engine(
    #we can override the LLM here
)

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions"""
    response = await query_engine.aquery(query)
    return str(response)

# BUILD THE AGENT
agent = AgentWorkflow.from_tools_or_functions(
    [search_documents], #we can have the agent do multiple functionalities, 
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can search through a 
    codebase and answer relavent questions with deep analysis""",
)

#----------------------------------
# Load FAISS Index Later
# this is a local index by Meta
#----------------------------------
def load_faiss_index() -> VectorStoreIndex:
    """Loads FAISS index from storage."""
    vector_store = FaissVectorStore.from_persist_dir(persist_dir=faissIndexPath)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )




# Load the FAISS index
index = load_faiss_index()
query_engine = index.as_query_engine(llm=Settings.llm)



# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run(
        "What did the author do in college? Also, what's 7 * 8?"
    )
    print(response)

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())