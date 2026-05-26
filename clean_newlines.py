import sqlite3
import pandas as pd
from pathlib import Path
import re

base_dir = Path(r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd')
friend_db = base_dir / 'export_for_friend' / 'fathomnet_openai_only.db'
main_db = base_dir / 'fathomnet_captioned' / 'fathomnet_with_captions.db'

def clean_newlines(text):
    if not isinstance(text, str): return text
    # Replace newlines and carriage returns with space
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Collapse multiple spaces into one and strip
    return re.sub(' +', ' ', text).strip()

def process_db(db_path):
    if not db_path.exists():
        print(f"File not found: {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT pair_id, caption FROM image_text_pairs', conn)
    
    count = 0
    for idx, row in df.iterrows():
        old_cap = row['caption']
        if not old_cap: continue
        new_cap = clean_newlines(old_cap)
        if new_cap != old_cap:
            conn.execute('UPDATE image_text_pairs SET caption = ? WHERE pair_id = ?', (new_cap, row['pair_id']))
            count += 1
            
    conn.commit()
    conn.close()
    print(f"Fixed {count} broken newlines in {db_path.name}")

# Process both DBs
process_db(friend_db)
process_db(main_db)

# Re-export CSV
conn = sqlite3.connect(friend_db)
df_final = pd.read_sql('''
    SELECT 
        i.filename AS file_name,
        p.caption AS text
    FROM image_text_pairs p
    JOIN images i ON p.image_id = i.image_id
    WHERE p.caption IS NOT NULL AND p.caption != ''
''', conn)
out_csv = base_dir / 'export_for_friend' / 'clip_kd_dataset.csv'
df_final.to_csv(out_csv, index=False)
conn.close()

print('Re-exported clean CSV successfully without newlines!')
