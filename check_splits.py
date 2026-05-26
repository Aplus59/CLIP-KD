import sqlite3
import pandas as pd
from pathlib import Path

db_path = r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd\db\fathomnet_cap.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql('SELECT split, COUNT(*) as count FROM images GROUP BY split', conn)
print("Splits in DB:")
print(df)
conn.close()
