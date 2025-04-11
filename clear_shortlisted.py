import sqlite3

conn = sqlite3.connect("database/job_match.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM Shortlisted")
conn.commit()
conn.close()

print("✅ Shortlisted table cleared successfully.")
