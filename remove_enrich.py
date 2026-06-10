import json
from pathlib import Path

nb_path = Path(r'c:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\notebooks\01_fathomnet_download_and_build_db.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
skip_next = False

for cell in nb['cells']:
    if skip_next:
        skip_next = False
        continue
        
    # Check if this is the enrich markdown cell
    is_enrich_header = False
    if cell['cell_type'] == 'markdown':
        source = "".join(cell.get('source', []))
        if '3. Enrich existing files' in source:
            is_enrich_header = True
            
    if is_enrich_header:
        skip_next = True # Skip the code cell that follows it
        continue
        
    # If not skipping, add to new_cells
    new_cells.append(cell)

# Update cell numbering for subsequent markdown cells
for cell in new_cells:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell.get('source', []))
        if '## 4. Build SQLite database' in source:
            cell['source'] = ['## 3. Build SQLite database']
        elif '## 5. Save CSV' in source:
            cell['source'] = ['## 4. Save CSV + summary']
        elif '## 6. Cleanup' in source:
            cell['source'] = ['## 5. Cleanup']

nb['cells'] = new_cells

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully removed enrichment cells and updated notebook numbering!")
