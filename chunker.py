import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
import hashlib
import json
import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def process_html_files(folder_path):
    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Filter out HTML files
    html_files = [file for file in all_files if file.endswith('.html')]

    # Initialize tokenizer and text_splitter
    tokenizer = tiktoken.get_encoding('cl100k_base')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )

    documents = []

    # Process each HTML file
    for html_file in tqdm(html_files):
        try:
            file_path = os.path.join(folder_path, html_file)

            # Load the HTML content
            with open(file_path, 'r') as f:
                content = f.read()

            # Generate a unique ID based on the file path
            m = hashlib.md5()
            m.update(file_path.encode('utf-8'))
            uid = m.hexdigest()[:12]

            # Split the content into chunks
            chunks = text_splitter.split_text(content)

            # Create document data
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f'{uid}-{i}',
                    'pageContent': chunk, # Use the key 'pageContent' instead of 'text'
                    'metadata': {
                        'txtPath': file_path  # Store the file path as 'txtPath' in metadata
                        # You can add other metadata fields if needed, similar to how 'loc' is in TypeScript
                    }
                })

            # Delete the HTML file after processing
            # os.remove(file_path)

        except Exception as e:
            print(f"Error processing file {html_file}: {e}")

    # Save the documents to a JSONL file
    with open('train.jsonl', 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')

    return documents

# Call the function with the folder path "websites"
folder_path = "websites"
documents = process_html_files(folder_path)
