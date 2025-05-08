import os
import re
import json
import time
import torch
import torch.nn.functional as F
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from pinecone import Pinecone

#HuggingFace URL
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Ollama API endpoint
OLLAMA_MODEL = "llama3.2"  # Locally served Ollama model name

PINECONE_API_KEY = "pcsk_5HvX5m_5T2tbcKPcp7P8pDhADq99KmQZ5JC2J27m2kUjzZwBQDomW6AmE3XJkCFMh8xV3d"

#babservice - Pinecone Index Variables & Directory Strings
INDEX_NAME = "mxbai-asls-index"
NAMESPACE = "ns1" #us-east-1

#bioapp - Pinecone Index Variables & Directory Strings
#INDEX_NAME = "ba-asls-index"
#NAMESPACE = "ns1" #us-east-1

#pnpservice - Pinecone Index Variables & Directory Strings
#INDEX_NAME = "pnp-asls-index"
#NAMESPACE = "ns1" #us-east-1

DEFAULT_FUNCTIONS_COUNT = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model gets initialized in gradio to stop it 
# from having to reinitialize every prompt
model = None

# -------------------
# Load Model Once
# -------------------
def initialize_model():
    global model
    print("Loading model...")
    dimensions = 512
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)
    print("Model loaded.")

#Generate an embedding for a single query string.
def generate_embedding(query):
    if not isinstance(query, str):
        raise ValueError("Input must be a string.")

    print("Generating embedding for query...")
    start_time = time.time()
    
    # Encode the query (no list wrapping needed)
    embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    
    elapsed_time = time.time() - start_time
    print(f"✔ Query embedding generated in {elapsed_time:.2f} seconds")

    return embedding.tolist()  # Convert tensor to list


# -------------------
# Query Pinecone
# -------------------    
def query_pinecone(embedded_query, top_k=4):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    try:
        results = index.query(
            vector=embedded_query,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE,
        )
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None

# -------------------
# Query Ollama Locally
# -------------------
def query_ollama(prompt, history, model=OLLAMA_MODEL):
    
    """
    Takes a list of tuples ([(user, bot), ...]) + a new 'prompt',
    converts it to Ollama's expected [{"role":..., "content":...}] format,
    sends it to Ollama, then returns the bot's reply as a string.
    """
    
    # 1) Convert from Gradio's tuple format -> Ollama's dictionary format
    ollama_formatted_history = []
    for (user_msg, bot_msg) in history:
        # user_msg => "user"
        ollama_formatted_history.append({"role": "user", "content": user_msg})
        # bot_msg => "system"
        ollama_formatted_history.append({"role": "assistant", "content": bot_msg})

    #ensure system instructions are in
    if not any(msg["role"] == "system" for msg in ollama_formatted_history):
        ollama_formatted_history.insert(0, {
            "role": "system",
            "content": f"""You are an AI assistant that remembers past conversations. You should always consider 
                            previous messages in the chat history to provide a coherent response. If a user references something 
                            earlier in the conversation, retrieve it from context and respond appropriately."""
        })

    # Append new prompt with context
    ollama_formatted_history.append({"role": "user", "content": prompt})

    
    payload = {
        "model": model,
        "messages": ollama_formatted_history,  # Include message history in the API call
        "options": {
            "temperature": 0.5,  
            "top_p": 0.8,  
            "max_tokens": 6000,  
            "repeat_penalty": 1.3,  
            "frequency_penalty": 0.3,  
            "presence_penalty": 0.2  
        }
    }

    try:
        # Enable streaming
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()  # Raise HTTP errors

        # Process the streamed response
        content = ""
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:  # Skip empty lines
                try:
                    # Parse each chunk as JSON
                    data = json.loads(chunk)
                    # Append content from the "content" key
                    if "message" in data and "content" in data["message"]:
                        content += data["message"]["content"]
                except json.JSONDecodeError:
                    pass

        print (content) if content else "NO CONTENT RECIEVED"

        return content if content else "[No content received]"
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return "[Error in Ollama response]"

# -------------------
# Extract Function Code & Comments
# -------------------
def get_function_code_and_comments(file_path, function_name):
    """
    Extracts both the function code and preceding comments.
    """
    if not os.path.exists(file_path):
        return "[File not found]", "[No comments found]"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin1") as f:
            content = f.read()

    # Extract function name without file prefix
    raw_function_name = function_name.split("::")[0] if "::" in function_name else function_name
    raw_function_name = raw_function_name.replace("()", "").strip()

    # Regex to find function definition
    function_pattern = rf"""
        (?:(public|private|protected|internal|static|async|override)\s+)*  
        (\w+(?:<[^>]+>)?)\s+           
        ({re.escape(raw_function_name)})\s*                       
        \(([^)]*)\)\s*                 
        (\{{|\n\s*\{{)                   
    """

    match = re.search(function_pattern, content, re.VERBOSE)
    if not match:
        return "[Function not found in file]", "[No comments found]"

    start_index = match.start()
    brace_count = 0
    function_body = []
    inside_function = False

    # Extract function body
    for char in content[start_index:]:
        if char == '{':
            brace_count += 1
            inside_function = True
        elif char == '}':
            brace_count -= 1
        
        function_body.append(char)
        if inside_function and brace_count == 0:
            break

    # Extract comments preceding the function
    pre_function_text = content[:start_index].splitlines()[-10:]  # Look at last 10 lines before function
    comments = []
    for line in reversed(pre_function_text):
        line = line.strip()
        if line.startswith("//") or line.startswith("/*"):
            comments.append(line.lstrip("/*").lstrip("//").strip())
        elif line == "":
            continue
        else:
            break  # Stop if there's a non-comment line

    comments.reverse()  # Restore comment order
    comment_text = " ".join(comments) if comments else "[No comments found]"

    return "".join(function_body), comment_text


# -------------------
# Generate Missing Comments
# -------------------
def generate_missing_comment_with_ollama(code_snippet, history):
    """
    Uses Ollama to generate a function description if none exists.
    """
    prompt = (
        f"### Code Snippet (C#)\n"
        f"```csharp\n{code_snippet}\n```\n\n"
        f"#### Task\n"
        f"Write a **detailed function comment** explaining what this function does. "
        f"Include a **high-level summary**, a list of **parameters and their purposes**, and a **description of the return value**. "
        f"Format the comment **professionally**, as if it were written by an experienced developer. "
        f"Use Markdown formatting and include bullet points for clarity."
    )

    return query_ollama(prompt, history)

# -------------------
# Generate Final Answer
# -------------------
def generate_final_answer_with_ollama(code_description, instructions, code_context_list, history):
    """
    Uses Ollama to generate a contextualized answer based on:
    - the code description (for context),
    - additional instructions (highest priority),
    - and the retrieved code snippets and their comments.
    """
    code_blocks = ""
    for i, item in enumerate(code_context_list, start=1):
        code_blocks += (
            f"Snippet {i} - Function: {item['function_full']}\n"
            f"Comment: {item['comment']}\n"
            f"Code Snippet:\n{item['snippet']}\n\n"
        )
    prompt = (
        f"## Instructions\n"
        f"{instructions}\n\n"
        f"## Code Description\n"
        f"{code_description}\n\n"
        f"## Relevant Code Snippets\n"
        f"{code_blocks}\n"
        f"### Task\n"
        f"Generate a **detailed, structured response** that fully explains the answer to the query. "
        f"Format the response in **Markdown**, using headings (`#`), subheadings (`##`), lists, and code blocks where needed. "
        f"Make sure to **break down complex explanations into steps** and provide examples where applicable."
    )
    return query_ollama(prompt, history)

def process_query(search_query, instructions, num_functions, conversation_mode, history):
    start_time = time.time()

    # If conversation_mode is enabled, we skip code search.
    if conversation_mode:
        answer = query_ollama(search_query, history)
        end_time = time.time()
        processing_time = end_time - start_time
        answer += f"\n\n**Processing Time:** {processing_time:.2f} seconds"
        return answer, ""

    try:
        top_k = int(num_functions)
    except ValueError:
        top_k = DEFAULT_FUNCTIONS_COUNT

    query_embedding = generate_embedding(search_query)
    if not query_embedding:
        return "Error generating query embedding.", ""

    results = query_pinecone(query_embedding, top_k=top_k)
    
    if results and "matches" in results:
        code_context_list = []
        
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            file_path = metadata.get("file_path", "Unknown")
            raw_function_name = metadata.get("function_name", "Unknown")
            search_function_name = raw_function_name.split("::")[-1] if "::" in raw_function_name else raw_function_name
            
            snippet, comment = get_function_code_and_comments(file_path, search_function_name)
            
            if comment == "[No comments found]":
                comment = generate_missing_comment_with_ollama(snippet, history)
            
            code_context_list.append({
                "function_full": raw_function_name,
                "snippet": snippet,
                "comment": comment,
                "file_path": file_path
            })
        
        # Generate a final detailed answer based on code context.
        final_answer = generate_final_answer_with_ollama(search_query, instructions, code_context_list, history)

        # Build a concise listing for the right-hand column.
        listing = "## 🔍 Top Matched Functions\n\n"
        listing += "| # | Function Name |\n|---|----------------|\n"
        for i, item in enumerate(code_context_list, start=1):
            listing += f"| {i} | `{item['function_full']}` |\n"

        listing += "\n## 📜 Function Details\n\n"
        for i, item in enumerate(code_context_list, start=1):
            listing += (
                f"### {i}. `{item['function_full']}`\n"
                f"📁 **File Path:** `{item['file_path']}`\n\n"
                f"**Comment:**\n> {item['comment']}\n\n"
                f"```csharp\n{item['snippet']}\n```\n\n"
                "---\n"
            )

        end_time = time.time()
        processing_time = end_time - start_time
        final_answer += f"\n\n**Processing Time:** {processing_time:.2f} seconds"

        return final_answer, listing
    else:
        return "No relevant results found in Pinecone.", ""

