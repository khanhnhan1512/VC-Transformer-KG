"""
Simple script to update OPENKE_file/entities.txt from OPENKE_file/entity2id.txt
Usage: python tools/update_entities_from_entity2id.py
"""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OPENKE = BASE / 'OPENKE_file'
IN_FILE = OPENKE / 'entity2id.txt'
OUT_FILE = OPENKE / 'entities.txt'

lines = []
with IN_FILE.open('r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        if i == 0:
            # first line may be the count; skip if numeric
            try:
                _ = int(line)
                continue
            except ValueError:
                pass
        # entity and id separated by whitespace or tab; take first column
        parts = line.split()
        if not parts:
            continue
        entity = parts[0].strip()
        # normalize: keep as-is (file loader lowercases) but strip spaces
        lines.append(entity)

OUT_FILE.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
print(f'Wrote {len(lines)} entries to {OUT_FILE}')