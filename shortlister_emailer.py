import os
import fitz  # PyMuPDF
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========== CONFIG ==========
RESUME_FOLDER = "data/resumes"
DATABASE_PATH = "database/job_match.db"
MATCH_THRESHOLD = 65  # Set your threshold here
# ============================

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to DB
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Load all Job Descriptions and their embeddings
cursor.execute("SELECT id, title, embedding FROM JobDescriptions")
jd_records = cursor.fetchall()

jd_embeddings = []
jd_ids = []
jd_titles = []

for jd_id, title, emb_blob in jd_records:
    embedding = np.frombuffer(emb_blob, dtype=np.float32)
    jd_embeddings.append(embedding)
    jd_ids.append(jd_id)
    jd_titles.append(title)

# Process all resumes
for filename in os.listdir(RESUME_FOLDER):
    if filename.endswith(".pdf"):
        filepath = os.path.join(RESUME_FOLDER, filename)

        # Skip already processed resumes
        cursor.execute("SELECT resume_name FROM Shortlisted WHERE resume_name = ?", (filename,))
        if cursor.fetchone():
            print(f"🔁 Skipping already processed: {filename}")
            continue

        # Extract resume text
        try:
            doc = fitz.open(filepath)
            resume_text = ""
            for page in doc:
                resume_text += page.get_text()
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue

        # Embed resume and compare
        resume_embedding = model.encode(resume_text).astype(np.float32).reshape(1, -1)
        similarities = cosine_similarity(resume_embedding, np.vstack(jd_embeddings))[0]

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx] * 100)
        jd_id = jd_ids[best_idx]
        jd_title = jd_titles[best_idx]

        print(f"{filename} matched '{jd_title}' with score: {best_score:.2f}%")

        if best_score >= MATCH_THRESHOLD:
            email_body = f"""
Dear Candidate,

We were impressed with your profile and would like to invite you for an interview for the position of "{jd_title}".

Please let us know your availability for an interview this week.

Best regards,  
Recruitment Team
            """.strip()

            cursor.execute('''
                INSERT INTO Shortlisted (resume_name, jd_id, score, email)
                VALUES (?, ?, ?, ?)
            ''', (filename, jd_id, best_score, email_body))

            print(f"✅ Shortlisted: {filename} → {jd_title} ({best_score:.2f}%)")
        else:
            print(f"⛔ Not shortlisted: Score below {MATCH_THRESHOLD}%")

# Finalize and close
conn.commit()
conn.close()
print("🎉 Done! All new resumes processed.")
