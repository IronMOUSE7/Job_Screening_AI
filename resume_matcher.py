import os
import sqlite3
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Init model
model = SentenceTransformer("all-MiniLM-L6-v2")

# PDF resume folder
resume_dir = "data/resumes"

# Connect to JD database
conn = sqlite3.connect("database/job_match.db")
cursor = conn.cursor()
cursor.execute("SELECT id, title, embedding FROM JobDescriptions")
jd_records = cursor.fetchall()

# Convert JD embeddings to usable format
jd_embeddings = []
jd_titles = []
jd_ids = []
for jd_id, title, emb_blob in jd_records:
    emb_vector = np.frombuffer(emb_blob, dtype=np.float32)
    jd_embeddings.append(emb_vector)
    jd_titles.append(title)
    jd_ids.append(jd_id)

jd_embeddings = np.vstack(jd_embeddings)

# Parse each resume
for filename in os.listdir(resume_dir):
    if filename.endswith(".pdf"):
        filepath = os.path.join(resume_dir, filename)
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()

        # Generate resume embedding
        resume_embedding = model.encode(text).reshape(1, -1)

        # Compare with all JDs
        similarities = cosine_similarity(resume_embedding, jd_embeddings)[0]

        # Show top match
        top_idx = np.argmax(similarities)
        score = similarities[top_idx] * 100
        print(f"\n📄 Resume: {filename}")
        print(f"🔗 Best match: {jd_titles[top_idx]}")
        print(f"✅ Match score: {score:.2f}%")
