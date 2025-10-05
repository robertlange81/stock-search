1. dcu
2. python3 -m venv .venv, dann python flask_app/app.py
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask transformers qdrant-client pillow

3. source .venv/bin/activate
4. python image_import_qdrant.py
5. curl -X POST -F "file=@/home/robert-lange/stock-search/testbilder/fliegenpilz.jpg"      http://127.0.0.1:5000/search