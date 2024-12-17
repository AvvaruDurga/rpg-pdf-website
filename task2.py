import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Function to scrape content from a URL
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return text

# Function to chunk text
def chunk_text(text, max_length=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

# Main scraping and embedding process
def main(urls):
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_chunks = []
    all_embeddings = []

    for url in urls:
        print(f"Scraping {url}...")
        text_content = scrape_website(url)
        chunks = chunk_text(text_content)
        all_chunks.extend(chunks)
        
        # Get embeddings for these chunks
        embeddings = embedding_model.encode(chunks)
        all_embeddings.append(embeddings)
    
    # Convert list of embeddings into a 2D numpy array
    all_embeddings = np.vstack(all_embeddings)

    # Initialize FAISS index
    index = faiss.IndexFlatL2(all_embeddings.shape[1])  # use L2 distance
    index.add(all_embeddings)  # Add embeddings to the index

    return all_chunks, index

# Example URLs
urls = [
    "https://www.uchicago.edu/",
    "https://www.washington.edu/",
    "https://www.stanford.edu/",
    "https://und.edu/"
]

chunks, index = main(urls)
print("Data ingestion complete.")

def handle_query(query, index, all_chunks):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode(query).reshape(1, -1)

    # Perform similarity search
    D, I = index.search(query_embedding, k=5)  # k is the number of nearest neighbors
    return [(I[0][i], D[0][i]) for i in range(len(I[0]))]

# Example Query
user_query = "What uw medicine are available?"
results = handle_query(user_query, index, chunks)

# Display results
for idx, distance in results:
    print(f"Result Chunk: {chunks[idx]}\nDistance: {distance}\n")