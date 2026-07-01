import json
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

print("Loading catalog...")

with open("data/raw/catalog_raw.json", "r", encoding="utf-8") as f:
    catalog = json.load(f, strict=False)

documents = []

for item in catalog:

    text = f"""
    Name: {item.get('name', '')}

    Description: {item.get('description', '')}

    Skills: {' '.join(item.get('keys', []))}

    Job Levels: {' '.join(item.get('job_levels', []))}

    Languages: {' '.join(item.get('languages', []))}

    Remote: {item.get('remote', '')}

    Adaptive: {item.get('adaptive', '')}
    """

    documents.append(text)

print(f"Loaded {len(documents)} assessments")

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

print("Building FAISS index...")

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, "vectorstore/shl.index")

with open("vectorstore/catalog.pkl", "wb") as f:
    pickle.dump(catalog, f)

print("\nDone!")
print("Total assessments:", len(catalog))