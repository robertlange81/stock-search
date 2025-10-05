import os
from PIL import Image
from flask import Flask, request, jsonify

import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

# ----------------------------
# Setup
# ----------------------------
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

client = QdrantClient("http://localhost:6333") # qdrant service from docker-compose
COLLECTION_NAME = "images"

def get_image_embedding(img_path: str):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze().cpu().numpy()

# ----------------------------
# Routes
# ----------------------------
@app.route("/search", methods=["POST"])
def search_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    tmp_path = "/tmp/query.jpg"
    file.save(tmp_path)

    try:
        q_emb = get_image_embedding(tmp_path)
        results = client.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=5)

        out = []
        for r in results:
            out.append({
                "label": r.payload.get("label", "unknown"),
                "filename": r.payload.get("filename", "unknown"),
                "score": r.score
            })

        return jsonify(out)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
