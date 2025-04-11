import streamlit as st
import sqlite3
import pandas as pd
import os
import fitz  # PyMuPDF
import numpy as np
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page setup
st.set_page_config(page_title="HR Screening Dashboard", layout="wide")
st.title("🧠 AI-Powered HR Screening Tool")
st.markdown("Welcome, HR! View and manage shortlisted candidates based on AI match scores.")

# Upload resume (sidebar)
st.sidebar.header("📤 Upload New Resume")
uploaded_file = st.sidebar.file_uploader("Upload a resume (PDF)", type=["pdf"])

# Connect to DB
conn = sqlite3.connect("database/job_match.db")
cursor = conn.cursor()

# Handle resume upload
if uploaded_file:
    save_path = os.path.join("data", "resumes", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")

    # Extract resume text
    resume_text = ""
    try:
        doc = fitz.open(save_path)
        for page in doc:
            resume_text += page.get_text()
    except Exception as e:
        st.sidebar.error(f"Error reading PDF: {e}")

    # Load JD embeddings
    cursor.execute("SELECT id, title, embedding FROM JobDescriptions")
    jd_records = cursor.fetchall()
    jd_embeddings = []
    jd_ids = []
    jd_titles = []

    for jd_id, title, emb_blob in jd_records:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        jd_embeddings.append(emb)
        jd_ids.append(jd_id)
        jd_titles.append(title)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_embedding = model.encode(resume_text).astype(np.float32).reshape(1, -1)
    similarities = cosine_similarity(resume_embedding, np.vstack(jd_embeddings))[0]

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx] * 100)
    jd_id = jd_ids[best_idx]
    jd_title = jd_titles[best_idx]

    st.sidebar.markdown("### 🔍 Match Result")
    st.sidebar.markdown(f"📄 Resume: `{uploaded_file.name}`")
    st.sidebar.markdown(f"🧠 Closest JD: `{jd_title}`")
    st.sidebar.markdown(f"✅ Score: `{best_score:.2f}%`")

    if best_score >= 80:
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
        ''', (uploaded_file.name, jd_id, best_score, email_body))
        conn.commit()
        st.sidebar.success("✅ Candidate shortlisted and added to dashboard!")
    else:
        st.sidebar.info("ℹ️ Resume not shortlisted (score below 80%).")

# Load shortlisted candidates
query = """
SELECT S.id, S.resume_name, S.score, S.email, JD.title AS jd_title
FROM Shortlisted S
JOIN JobDescriptions JD ON S.jd_id = JD.id
ORDER BY S.score DESC
"""
df = pd.read_sql_query(query, conn)

if df.empty:
    st.warning("⚠️ No candidates shortlisted yet. Run the shortlister script or upload resumes.")
else:
    st.subheader("📋 Shortlisted Candidates")

    # Filter by JD title
    unique_jds = df["jd_title"].unique().tolist()
    selected_jd = st.selectbox("🧾 Filter by Job Title:", ["All"] + unique_jds)

    if selected_jd != "All":
        filtered_df = df[df["jd_title"] == selected_jd]
    else:
        filtered_df = df

    # Match Score Display (Visual)
    st.markdown("### 🎯 Candidates Matching This Role")
    for _, row in filtered_df.iterrows():
        col1, col2, col3 = st.columns([2, 2, 4])
        with col1:
            st.markdown(f"**🧑 Resume:** `{row['resume_name']}`")
        with col2:
            st.markdown(f"**📌 JD:** `{row['jd_title']}`")
        with col3:
            percent = float(row['score'])
            if percent >= 80:
                color = "green"
            elif percent >= 60:
                color = "orange"
            else:
                color = "red"
            st.markdown(
                f"""<div style="font-size:16px;">
                    Match Score: <span style='color:{color}; font-weight:bold'>{percent:.2f}%</span>
                </div>""",
                unsafe_allow_html=True
            )
        st.markdown("---")

    # Export to Excel
    if not filtered_df.empty:
        export_df = filtered_df.rename(columns={
            "resume_name": "Resume",
            "jd_title": "Matched JD",
            "score": "Score",
            "email": "Email"
        })[["Resume", "Matched JD", "Score", "Email"]]

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Shortlisted")

        st.download_button(
            label="📥 Export Shortlist to Excel",
            data=buffer.getvalue(),
            file_name="shortlisted_candidates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Candidate selection and email preview
    candidate_names = filtered_df["resume_name"].tolist()
    if candidate_names:
        selected_resume = st.selectbox("📄 Select a candidate to view details:", candidate_names)
        selected_row = filtered_df[filtered_df["resume_name"] == selected_resume].iloc[0]

        st.markdown(f"### ✉️ Interview Email for `{selected_resume}`")
        st.text_area("Email Content", value=selected_row["email"], height=300)
        st.success(f"Candidate matched for: {selected_row['jd_title']} with {float(selected_row['score']):.2f}% match.")

        # Resume Viewer
        with st.expander("📄 View Resume Text"):
            resume_path = os.path.join("data", "resumes", selected_resume)
            resume_text = ""
            try:
                doc = fitz.open(resume_path)
                for page in doc:
                    resume_text += page.get_text()
                st.text_area("Extracted Resume Content", resume_text, height=400)
            except Exception as e:
                st.error(f"Error reading resume file: {e}")

        # Simulated Email Confirmation
        if st.button("📨 Send Interview Email"):
            st.success("✅ Simulated: Email has been sent successfully to the candidate.")

conn.close()
