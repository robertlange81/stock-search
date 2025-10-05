1. dcu
2. python flask_app/app.py
3. source .venv/bin/activate
4. python image_import_qdrant.py
5. curl -X POST -F "file=@/home/robert-lange/stock-search/testbilder/fliegenpilz.jpg"      http://127.0.0.1:5000/search