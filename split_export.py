import sqlite3
import pandas as pd
from pathlib import Path

base_dir = Path(r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd\db')
db_path = base_dir / 'fathomnet_cap.db'

conn = sqlite3.connect(db_path)
df_final = pd.read_sql('''
    SELECT 
        i.filename AS file_name,
        p.caption AS text,
        i.split
    FROM image_text_pairs p
    JOIN images i ON p.image_id = i.image_id
    WHERE p.caption IS NOT NULL AND p.caption != ''
''', conn)

# Export individual splits
for split_name in df_final['split'].unique():
    if not split_name: continue
    split_df = df_final[df_final['split'] == split_name][['file_name', 'text']]
    out_csv = base_dir / f'{split_name}.csv'
    split_df.to_csv(out_csv, index=False)
    print(f'Exported {split_name}.csv with {len(split_df)} rows.')

conn.close()
