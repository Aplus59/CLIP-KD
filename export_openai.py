import sqlite3
import pandas as pd
import shutil
from pathlib import Path

base_dir = Path(r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd')
origin_db = base_dir / 'fathomnet_captioned' / 'fathomnet_with_captions.db'

# Create export folder
export_dir = base_dir / 'export_for_friend'
export_dir.mkdir(exist_ok=True)

# 1. Create clean DB
clean_db = export_dir / 'fathomnet_openai_only.db'
if clean_db.exists():
    clean_db.unlink()
shutil.copy2(origin_db, clean_db)

conn = sqlite3.connect(clean_db)
c = conn.cursor()
# Delete all rows that are NOT gpt-4o-mini (e.g. BLIP)
c.execute("DELETE FROM image_text_pairs WHERE model_name != 'gpt-4o-mini' AND model_name IS NOT NULL")
conn.commit()

# 2. Export clean CSV
df_final = pd.read_sql('''
    SELECT 
        i.filename AS file_name,
        p.caption AS text
    FROM image_text_pairs p
    JOIN images i ON p.image_id = i.image_id
    WHERE p.caption IS NOT NULL AND p.caption != ''
''', conn)

out_csv = export_dir / 'clip_kd_dataset.csv'
df_final.to_csv(out_csv, index=False)
conn.close()

print(f'Done! Clean DB and CSV are in {export_dir}')
