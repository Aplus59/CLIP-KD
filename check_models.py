import sqlite3
import pandas as pd

db_path = r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd\fathomnet_captioned\fathomnet_with_captions.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql('SELECT DISTINCT model_name, COUNT(*) as count FROM image_text_pairs GROUP BY model_name', conn)
print(df.to_string(index=False))
conn.close()
