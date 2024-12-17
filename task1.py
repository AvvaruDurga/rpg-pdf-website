import pdfplumber  # For PDF extraction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load PDF and extract text
pdf_path = "./static/rpg.pdf"  # <-- Change this to your PDF file path

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

with pdfplumber.open(pdf_path) as pdf:
    # Extract text from page 2 (usually index 1 in PDF)
    page_2_text = pdf.pages[1].extract_text()
    
    # Extract all tables from page 6 (usually index 5 in PDF)
    page_6_tables = pdf.pages[5].extract_tables()

# Create embeddings for page 2 text and tables from page 6
combined_data = [page_2_text]  # Convert each table to string
for table in page_6_tables:
    if isinstance(table, list) and all(isinstance(item, str) for item in table):
        # Only join if all elements in the table are strings
        combined_data.append('\n'.join(table)) 
    else:
        # Handle a case where it isn't a valid table or log the issue
        print("Skipping invalid table format:", table)
embeddings = embedding_model.encode(combined_data)

# Here you would normally insert the embeddings into a vector database (e.g., FAISS)
# For now, let’s just print the embeddings
print("Embeddings created for the following entries:")
for i, entry in enumerate(combined_data):
    print(f"Entry {i} - Length: {len(entry)} characters")

# Example query
user_query = "What is the earnings and unemployment rates by educational attainment?"
query_embedding = embedding_model.encode(user_query)

# Calculate cosine similarities to fetch closest matches
similarities = cosine_similarity([query_embedding], embeddings)
closest_idx = np.argmax(similarities)  # Get the index with the highest similarity

# Output the most relevant chunk
print("\nMost relevant chunk from PDF:")
print(combined_data[closest_idx])