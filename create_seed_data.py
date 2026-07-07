import pandas as pd
import shutil
import os

# Paths
BASE_DIR = r"c:\Users\user\Desktop\HITL, AL, research"
METADATA_PATH = os.path.join(BASE_DIR, "Ham-1000000 Dataset for skin", "HAM10000_metadata.csv")
IMG_PART1 = os.path.join(BASE_DIR, "Ham-1000000 Dataset for skin", "HAM10000_images_part_1")
IMG_PART2 = os.path.join(BASE_DIR, "Ham-1000000 Dataset for skin", "HAM10000_images_part_2")
SEED_DIR = os.path.join(BASE_DIR, "Seed Data")
SEED_CSV = os.path.join(SEED_DIR, "seed_metadata.csv")
REMAINING_CSV = os.path.join(BASE_DIR, "Ham-1000000 Dataset for skin", "HAM10000_metadata_remaining.csv")

SAMPLES_PER_CLASS = 70
CLASSES = ['mel', 'bcc', 'akiec', 'nv', 'bkl', 'df', 'vasc']

# Read metadata
df = pd.read_csv(METADATA_PATH)
print(f"Total rows in original metadata: {len(df)}")
print(f"\nClass distribution:")
print(df['dx'].value_counts())

# Sample 70 per class
seed_frames = []
for cls in CLASSES:
    cls_df = df[df['dx'] == cls]
    count = len(cls_df)
    if count < SAMPLES_PER_CLASS:
        print(f"WARNING: class '{cls}' only has {count} samples, taking all of them.")
        sampled = cls_df
    else:
        sampled = cls_df.sample(n=SAMPLES_PER_CLASS, random_state=42)
    seed_frames.append(sampled)
    print(f"  {cls}: selected {len(sampled)} images")

seed_df = pd.concat(seed_frames, ignore_index=True)
print(f"\nTotal seed samples selected: {len(seed_df)}")

# Copy images to Seed Data folder
os.makedirs(SEED_DIR, exist_ok=True)
copied = 0
not_found = 0
for image_id in seed_df['image_id']:
    filename = f"{image_id}.jpg"
    src_part1 = os.path.join(IMG_PART1, filename)
    src_part2 = os.path.join(IMG_PART2, filename)
    dst = os.path.join(SEED_DIR, filename)

    if os.path.exists(src_part1):
        shutil.move(src_part1, dst)
        copied += 1
    elif os.path.exists(src_part2):
        shutil.move(src_part2, dst)
        copied += 1
    else:
        print(f"  IMAGE NOT FOUND: {filename}")
        not_found += 1

print(f"\nCopied {copied} images to Seed Data folder.")
if not_found:
    print(f"  {not_found} images were not found.")

# Save seed metadata CSV
seed_df.to_csv(SEED_CSV, index=False)
print(f"Seed metadata saved to: {SEED_CSV}")

# Remove seed rows from original and save remaining
remaining_df = df[~df['image_id'].isin(seed_df['image_id'])]
remaining_df.to_csv(REMAINING_CSV, index=False)
print(f"Remaining metadata saved to: {REMAINING_CSV}")
print(f"Remaining rows: {len(remaining_df)}")
