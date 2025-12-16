"""Debug PDF loading and chunking."""

import sys
sys.path.insert(0, ".")

from src.loaders.factory import LoaderFactory
from src.chunkers.factory import ChunkerFactory

# Load PDF
print("Loading PDF...")
docs = LoaderFactory.load("data/books/machine_learning_basics.pdf")
print(f"Loaded {len(docs)} documents")

# Show first doc
print(f"\nFirst doc metadata: {docs[0].metadata}")
print(f"First doc content (first 500 chars):")
print(repr(docs[0].content[:500]))

# Chunk
print("\n\nChunking...")
chunker = ChunkerFactory.from_config({
    "strategy": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 100
})

chunks = []
for doc in docs:
    chunks.extend(chunker.chunk(doc))

print(f"Created {len(chunks)} chunks")

# Show first chunk
print(f"\nFirst chunk content (first 300 chars):")
print(repr(chunks[0].content[:300]))

# Test embedding on first chunk
print("\n\nTesting embedding on first chunk...")
import ollama
client = ollama.Client(host="http://localhost:11434")

text = chunks[0].content[:500]  # Short test
print(f"Text length: {len(text)}")
print(f"Text preview: {repr(text[:100])}")

try:
    response = client.embeddings(model="nomic-embed-text", prompt=text)
    print(f"✓ Embedding succeeded! Length: {len(response['embedding'])}")
except Exception as e:
    print(f"✗ Embedding failed: {e}")
