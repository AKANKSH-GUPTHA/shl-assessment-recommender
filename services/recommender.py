import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("vectorstore/shl.index")

with open("vectorstore/catalog.pkl", "rb") as f:
    catalog = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")


def recommend_assessments(query, top_k=10):

    query = query.lower()

    if "java" in query:
        query += " core java spring sql"

    if "spring" in query:
        query += " spring framework microservices"

    if "aws" in query:
        query += " amazon web services cloud"

    if "react" in query:
        query += " frontend javascript react"

    if "python" in query:
        query += " python programming"

    embedding = model.encode([query])

    embedding = np.array(embedding).astype("float32")

    distances, indices = index.search(embedding, top_k)

    recommendations = []

    for idx in indices[0]:

        item = catalog[idx]

        name = item.get("name", "").lower()

        # Skip report products
        if "report" in name:
            continue

        recommendations.append({
            "name": item.get("name"),
            "url": item.get("link"),
            "test_type": ", ".join(item.get("keys", []))
        })

        if len(recommendations) == 5:
            break

    return recommendations