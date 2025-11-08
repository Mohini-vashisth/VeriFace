#!/usr/bin/env python3
import csv, os, shutil, random
from pathlib import Path

ROOT = Path("data/ffpp_c23")
OUT = Path("data/ffpp_c23_subset")
LABELS = ROOT / "labels.csv"
N_PER_CLASS = 50  # change to 20 or 10 if you want smaller

OUT_real = OUT / "real"
OUT_fake = OUT / "fake"
OUT_real.mkdir(parents=True, exist_ok=True)
OUT_fake.mkdir(parents=True, exist_ok=True)

rows = []
with LABELS.open() as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

reals = [r['video_path'] for r in rows if r['label'] == '0']
fakes = [r['video_path'] for r in rows if r['label'] == '1']

random.shuffle(reals)
random.shuffle(fakes)

reals_sel = reals[:N_PER_CLASS]
fakes_sel = fakes[:N_PER_CLASS]

def copy_list(lst, out_dir):
    for p in lst:
        src = Path(p)
        dst = out_dir / src.name
        if not dst.exists():
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print("FAILED copy:", src, e)

print("Copying real:", len(reals_sel), "fake:", len(fakes_sel))
copy_list(reals_sel, OUT_real)
copy_list(fakes_sel, OUT_fake)
print("Subset created at", OUT)
