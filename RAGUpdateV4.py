import os
import time
import re
import torch
import torch.nn.functional as F
from pinecone import Pinecone
from pathspec import PathSpec  # Library for handling .gitignore patterns
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings


PINECONE_API_KEY = "pcsk_5HvX5m_5T2tbcKPcp7P8pDhADq99KmQZ5JC2J27m2kUjzZwBQDomW6AmE3XJkCFMh8xV3d"


#babservice - Pinecone Index Variables & Directory Strings
#CODEBASE_PATH = r"C:\Users\ASLS\source\asls\bab"
#GITIGNORE_PATH = r"C:\Users\ASLS\source\asls\bab\.gitignore"
#INDEX_NAME = "mxbai-asls-index"
#NAMESPACE = "ns1" #us-east-1


#bioapp - Pinecone Index Variables & Directory Strings
CODEBASE_PATH = r"C:\Users\ASLS\source\asls\bioapp"
GITIGNORE_PATH = r"C:\Users\ASLS\source\asls\bioapp\.gitignore"
INDEX_NAME = "ba-asls-index"
NAMESPACE = "ns1" #us-east-1

#pnpservice - Pinecone Index Variables & Directory Strings
#CODEBASE_PATH = r"C:\Users\ASLS\source\asls\bioapp\bioapp-services\services\Asls.Hardware.PickAndPlace"  # Replace with the path to your codebase
#GITIGNORE_PATH = r"C:\Users\ASLS\source\asls\bioapp\bioapp-services\.gitignore"
#INDEX_NAME = "pnp-asls-index"
#NAMESPACE = "ns1" #us-east-1

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
device = torch.device ("cpu")
dimensions = 512
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

# ---------------------------
# Batch embedding generation with progress printing
# ---------------------------
def generate_embeddings(code_snippets, batch_size=8):
    """
    Generate embeddings for a batch of code snippets while tracking progress.
    Uses `SentenceTransformer.encode()` but retains manual progress updates.
    """
    embeddings = []
    total = len(code_snippets)

    for i in range(0, total, batch_size):
        batch = code_snippets[i : i + batch_size]
        remaining = total - i
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} functions (Remaining: {remaining})")

        start_time = time.time()  # Track time taken per batch
        batch_embeddings = model.encode(batch, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
        elapsed_time = time.time() - start_time

        print(f"✔ Batch {i//batch_size + 1} completed in {elapsed_time:.2f} seconds")

        embeddings.extend(batch_embeddings.tolist())

    return embeddings

        
#Reads the .gitignore file and generates a PathSpec object
#that can be used to check if files or directories match the ignore rules.
def load_gitignore_patterns(gitignore_path):
    
    try:
        with open(gitignore_path, "r") as f:
            patterns = f.readlines()
        return PathSpec.from_lines('gitwildmatch', patterns)
    except FileNotFoundError:
        print(f".gitignore file not found at {gitignore_path}. Proceeding without it.")
        return None


def clear_pinecone_index(index, namespace=""):
    """
    Clear all vectors in a given namespace within the Pinecone index.
    If no namespace is specified, clears the entire index.
    """
    try:
        print(f"Clearing Pinecone index. Namespace: '{namespace}'")
        index.delete(delete_all=True, namespace=namespace)
        print("Pinecone index cleared.")
    except Exception as e:
        print(f"Error clearing Pinecone index: {e}")


def upsert_to_pinecone(index, vectors, namespace, chunk_size=100):
    """
    Upsert vectors to Pinecone in smaller chunks to avoid size limit errors.
    Args:
        index: Pinecone index object.
        vectors: List of vectors to upsert.
        namespace: Namespace for the Pinecone index.
        chunk_size: Number of vectors per batch.
    """
    total_vectors = len(vectors)
    for i in range(0, total_vectors, chunk_size):
        batch = vectors[i:i + chunk_size]
        try:
            print(f"Upserting batch {i//chunk_size + 1}/{(total_vectors + chunk_size - 1)//chunk_size} to Pinecone...")
            index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            print(f"Error upserting batch {i//chunk_size + 1}: {e}")

# Extract all functions from a given file
def extract_functions_from_file(file_path):

    try:
        # Attempt to read the file using UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except UnicodeDecodeError:
        try:
            # Fallback to a more permissive encoding
            with open(file_path, "r", encoding="latin1") as f:
                file_content = f.read()
        except Exception as e:
            print(f"Failed to read file {file_path} due to encoding issues: {e}")
            return []  # Skip this file

    try:
        # Define a list of reserved .NET keywords to ignore
        reserved_keywords = {
            "foreach", "for", "while", "if", "else", "switch", "case", 
            "default", "return", "break", "continue", "try", "catch", 
            "finally", "using", "throw", "lock", "yield", "async", 
            "await", "do"
        }

        # Use a regular expression to extract functions from the C# file
        function_pattern = r"""
            (?:(public|private|protected|internal|static|async|override)\s+)*  # Optional modifiers
            (\w+(?:<[^>]+>)?)\s+           # Return type, including generics like Task<GetPathsResponse>
            (\w+)\s*                       # Function name
            \(([^)]*)\)\s*                 # Parameters
            (\{|\n\s*\{)                   # Function body start (allows for newline before opening brace)
        """
        functions = []
        for match in re.finditer(function_pattern, file_content, re.VERBOSE):

            function_name_no_file = match.group(3)

            # Skip reserved keywords
            if function_name_no_file in reserved_keywords:
                continue

            start = match.start()
            brace_count = 0
            function_body = []
            inside_function = False
            # Extract the complete function body using brace matching
            for char in file_content[start:]:
                if char == '{':
                    brace_count += 1
                    inside_function = True
                elif char == '}':
                    brace_count -= 1
                function_body.append(char)
                if inside_function and brace_count == 0:
                    break

            # Extract comments or docstrings preceding the function
            pre_function_text = file_content[:start].splitlines()[-10:]  # Last 10 lines before function
            comments = []
            for line in reversed(pre_function_text):
                line = line.strip()
                if line.startswith("//") or line.startswith("/*"):
                    comments.append(line.lstrip("/*").lstrip("//").strip())
                elif line == "":
                    continue
                else:
                    break
            comments.reverse()  # Restore original order of comments
            comment_text = " ".join(comments) if comments else None

            function_name = f"{os.path.basename(file_path)}::{match.group(3)}()"
            functions.append((function_name, "".join(function_body), comment_text))   
        return functions
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return []

# ---------------------------
# Main Function
# ---------------------------
if __name__ == "__main__":
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Load gitignore patterns
    gitignore_spec = load_gitignore_patterns(GITIGNORE_PATH)
    processed_files = []

    all_code_snippets = []
    all_function_metadata = []

    # Step 1: Process files in the codebase
    for root, _, files in os.walk(CODEBASE_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip files matching .gitignore patterns
            if gitignore_spec and gitignore_spec.match_file(os.path.relpath(file_path, CODEBASE_PATH)):
                continue

            # Process only .cs files
            if file.lower().endswith(".cs") and file_path not in processed_files:
                processed_files.append(file_path)

                functions = extract_functions_from_file(file_path)
                if functions:
                    print(f"Processing file: {file_path} -> Found {len(functions)} functions.")
                    for fn_name, fn_body, comment_text in functions:
                        all_code_snippets.append(fn_body)
                        metadata = {
                            "file_path": file_path,
                            "function_name": fn_name,
                            "comments": comment_text if comment_text else ""
                        }
                        all_function_metadata.append(metadata)
                else:
                    print(f"No recognized functions in {file_path}.")

    # Step 2: Generate embeddings and upsert to Pinecone
    if all_code_snippets:
        total_functions = len(all_code_snippets)
        print(f"Generating embeddings for {total_functions} functions.")
        embeddings = generate_embeddings(all_code_snippets, batch_size=8)

        embeddings_to_upsert = []
        for embedding, metadata in zip(embeddings, all_function_metadata):
            # Build a unique ID for each function
            vector_id = f"{metadata['file_path']}::{metadata['function_name']}"
            # Simplify metadata if needed
            upsert_metadata = {k: v for k, v in metadata.items() if k != "comments"}
            embeddings_to_upsert.append((vector_id, embedding, upsert_metadata))

        # Clear Pinecone index before upserting
        clear_pinecone_index(index, namespace=NAMESPACE)
        upsert_to_pinecone(index, embeddings_to_upsert, namespace=NAMESPACE, chunk_size=100)
    else:
        print("No functions found to process.")
   