import sqlite3
import pandas as pd

db_path = r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd\fathomnet_captioned\fathomnet_with_captions.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Fix Typos
c.execute("UPDATE images SET species_clean = 'Arthropoda' WHERE species_clean = 'Arhtropoda'")
c.execute("UPDATE images SET species_clean = 'Cnidaria' WHERE species_clean = 'Cnidarai'")
c.execute("UPDATE images SET species_clean = 'Echinodermata' WHERE species_clean = 'Echinoderma'")
conn.commit()

# Export clean CSV for friend
df_final = pd.read_sql('''
    SELECT 
        i.filename AS file_name,
        p.caption AS text
    FROM image_text_pairs p
    JOIN images i ON p.image_id = i.image_id
    WHERE p.caption IS NOT NULL AND p.caption != ''
''', conn)

out_csv = r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\newd\clip_kd_dataset.csv'
df_final.to_csv(out_csv, index=False)
conn.close()

print('Fixed typos and exported CSV successfully to:', out_csv)
