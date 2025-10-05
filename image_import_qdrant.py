import os
from pathlib import Path
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


# ----------------------------
# Settings
# ----------------------------
IMAGE_DIR = "./bilder"         # top-level folder with all categories
COLLECTION_NAME = "images"
QDRANT_URL = "http://localhost:6333"


# ----------------------------
# Load CLIP model (image embeddings)
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_embedding(img_path: str):
    """Convert image to embedding vector"""
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze().cpu().numpy()


# ----------------------------
# Connect to Qdrant
# ----------------------------
client = QdrantClient(QDRANT_URL)

# Create collection only if it doesn't exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )
    print(f"✅ Created new collection '{COLLECTION_NAME}'")
else:
    print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists – appending new images only.")


# ----------------------------
# Get already imported filenames
# ----------------------------
existing_points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=10000)
known_files = {p.payload.get("filename") for p in existing_points if "filename" in p.payload}
print(f"🔍 Found {len(known_files)} existing images in collection.")


# ----------------------------
# Import new images
# ----------------------------
points = []
idx = len(known_files) + 1

for img_file in Path(IMAGE_DIR).rglob("*.jpg"):
    if img_file.name in known_files:
        continue  # skip duplicates

    try:
        emb = get_image_embedding(str(img_file))
        # Use directory names as labels
        category = img_file.parts[-3] if len(img_file.parts) >= 3 else "unknown"
        label = img_file.parts[-2] if len(img_file.parts) >= 2 else "unknown"

        payload = {
            "filename": img_file.name,
            "category": category,   # e.g. "pilze"
            "label": label          # e.g. "steinpilz"
        }
        points.append(PointStruct(id=idx, vector=emb, payload=payload))
        print(f"➕ Imported new image: {img_file} → {label}")
        idx += 1
    except Exception as e:
        print(f"⚠️ Error with {img_file}: {e}")

if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\n✅ Imported {len(points)} new images into Qdrant collection '{COLLECTION_NAME}'.")
else:
    print("\nℹ️ No new images to import.")


# ----------------------------
# Example search with one query image
# ----------------------------
query_image = "./testbilder/query.jpg"
if os.path.exists(query_image):
    q_emb = get_image_embedding(query_image)
    results = client.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=5)
    print("\n🔎 Top results for query image:")
    for r in results:
        print(f"- {r.payload['label']} / {r.payload['filename']} (score={r.score:.4f})")
else:
    print("\nℹ️ No query image found at ./testbilder/query.jpg – add one to test search.")
