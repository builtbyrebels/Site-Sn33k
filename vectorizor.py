## pip install -U openai pinecone-client jsonlines
## set up pinecone database with 1536 dimensions
import jsonlines
import openai
from pinecone import Pinecone
import os 
import time
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

load_dotenv()
# Set up OpenAI and Pinecone API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
#PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")



# Load train.jsonl file
def load_data(file_path):
    data = []
    with jsonlines.open(file_path) as f:
        for item in f:
            data.append(item)
    return data

# Initialize OpenAI API
def init_openai(api_key):
    openai.api_key = api_key
    return "text-embedding-3-small"

# Initialize Pinecone index
def init_pinecone(api_key, index_name, dimension):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=dimension)
    return pc.Index(index_name)

# Create embeddings and populate the index
@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def create_embedding_with_retry(text_batch, model):
    return openai.Embedding.create(input=text_batch, engine=model)

def create_and_index_embeddings(data, model, index):
    count = 0
    batch_size = 64
    for start_index in range(0, len(data), batch_size):
        # Correctly use 'pageContent' instead of 'text'
        text_batch = [item["pageContent"] for item in data[start_index:start_index+batch_size]]
        # Correct the references for ids_batch based on the new structure
        ids_batch = [
            f"{item['metadata']['txtPath'].split('/')[-1]}_{i}"  # Use 'txtPath' from within 'metadata'
            for i, item in enumerate(data[start_index:start_index+batch_size])
        ]

        res = create_embedding_with_retry(text_batch, model)
        
        embeds = [record["embedding"] for record in res["data"]]
        # Update 'to_upsert' with the correct metadata structure
        to_upsert = [
            {
                "id": ids_batch[i],
                "values": embeds[i],
                "metadata": {
                    "txtPath": data[start_index + i]["metadata"]["txtPath"],
                    "pageContent": text_batch[i]
                }
            }
            for i in range(len(embeds))
        ]

        index.upsert(vectors=to_upsert)

       # Delete the last upserted entry
       # to_upsert.pop()

        count = count + 1
        print(count)
        #time.sleep(2)

if __name__ == "__main__":
    # Load the data from train.jsonl
    train_data = load_data("train.jsonl")

    # Initialize OpenAI Embedding API
    MODEL = init_openai(OPENAI_API_KEY)

    # Get embeddings dimension
    sample_embedding = openai.Embedding.create(input="sample text", engine=MODEL)["data"][0]["embedding"]
    EMBEDDING_DIMENSION = len(sample_embedding)

    # Initialize Pinecone index
    chatgpt_index = init_pinecone(PINECONE_API_KEY, INDEX_NAME, EMBEDDING_DIMENSION)

    # Create embeddings and populate the index with the train data
    
    create_and_index_embeddings(train_data, MODEL, chatgpt_index)
