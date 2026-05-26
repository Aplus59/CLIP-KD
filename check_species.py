import sqlite3
import pandas as pd

db_path = r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd\fathomnet_captioned\fathomnet_with_captions.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql('SELECT DISTINCT species_clean FROM images', conn)
print(df.sort_values(by='species_clean').to_string(index=False))
conn.close()
