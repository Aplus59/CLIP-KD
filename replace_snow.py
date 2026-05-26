import pandas as pd
import sqlite3
import re

db_path = r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd\fathomnet_captioned\fathomnet_with_captions.db'
csv_path = r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd\image_text_pairs.csv'

try:
    df = pd.read_csv(csv_path)
    count = df['caption'].str.contains('marine snow', case=False, na=False).sum()
    df['caption'] = df['caption'].str.replace(r'(?i)marine snow', 'salt-and-pepper dots', regex=True)
    df.to_csv(csv_path, index=False)
    print(f'CSV updated: {count} matches replaced.')

    conn = sqlite3.connect(db_path)
    df_db = pd.read_sql("SELECT pair_id, caption FROM image_text_pairs WHERE caption LIKE '%marine snow%'", conn)
    for idx, row in df_db.iterrows():
        new_caption = re.sub(r'(?i)marine snow', 'salt-and-pepper dots', str(row['caption']))
        conn.execute('UPDATE image_text_pairs SET caption = ? WHERE pair_id = ?', (new_caption, row['pair_id']))
    conn.commit()
    conn.close()
    print(f'Database updated: {len(df_db)} matches replaced.')
except Exception as e:
    print('Error:', e)
