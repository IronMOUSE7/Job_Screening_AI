import pandas as pd
import sqlite3
from sentence_transformers import SentenceTransformer
import os

# Load transformer model for semantic embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Load job descriptions from CSV
csv_path = "job_description.csv"
df = pd.read_csv(csv_path)

# Step 2: Create or connect to SQLite DB
db_path = "database/job_match.db"
os.makedirs("database", exist_ok=True)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 3: Create JD Table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS JobDescriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        jd_text TEXT,
        skills TEXT,
        experience TEXT,
        embedding BLOB
    )
''')

# Step 4: Process and store JDs
for _, row in df.iterrows():
    title = row.get("Job Title", "")
    jd_text = row.get("Job Description", "")
    skills = row.get("Skills", "")
    experience = row.get("Experience", "")

    # Combine content for embedding
    full_text = f"{title}. {jd_text}. Skills: {skills}. Experience: {experience}"
    embedding = model.encode(full_text).tobytes()

    cursor.execute('''
        INSERT INTO JobDescriptions (title, jd_text, skills, experience, embedding)
        VALUES (?, ?, ?, ?, ?)
    ''', (title, jd_text, skills, experience, embedding))

conn.commit()
conn.close()
print("✅ JD Summaries and embeddings stored in SQLite!")
