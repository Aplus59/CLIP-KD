import pandas as pd
import sqlite3
import re
from pathlib import Path

base_dir = Path(r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd')

for csv_path in base_dir.rglob('image_text_pairs.csv'):
    try:
        df = pd.read_csv(csv_path)
        count = df['caption'].str.contains('marine snow', case=False, na=False).sum()
        if count > 0:
            df['caption'] = df['caption'].str.replace(r'(?i)marine snow', 'salt-and-pepper dots', regex=True)
            df.to_csv(csv_path, index=False)
            print(f'Updated {csv_path.name} in {csv_path.parent.name}: {count} matches replaced.')
    except Exception as e:
        print(f'Error reading {csv_path}: {e}')

for db_path in base_dir.rglob('*.db'):
    try:
        conn = sqlite3.connect(db_path)
        df_db = pd.read_sql("SELECT pair_id, caption FROM image_text_pairs WHERE caption LIKE '%marine snow%'", conn)
        if len(df_db) > 0:
            for idx, row in df_db.iterrows():
                new_caption = re.sub(r'(?i)marine snow', 'salt-and-pepper dots', str(row['caption']))
                conn.execute('UPDATE image_text_pairs SET caption = ? WHERE pair_id = ?', (new_caption, row['pair_id']))
            conn.commit()
            print(f'Updated DB {db_path.name} in {db_path.parent.name}: {len(df_db)} matches replaced.')
        conn.close()
    except Exception as e:
        pass
print('Done!')
