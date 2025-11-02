# ========== STEP 1: Ensure dataset exists ==========
csv_path = "data/heart.csv"

# ✅ Verified working dataset source (UCI Heart Disease data)
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/HeartDisease.csv"

if not os.path.exists(csv_path):
    os.makedirs("data", exist_ok=True)
    print("📥 Downloading dataset from:", data_url)
    try:
        response = requests.get(data_url)
        response.raise_for_status()
        with open(csv_path, "wb") as f:
            f.write(response.content)
        print("✅ heart.csv downloaded successfully.")
    except Exception as e:
        print("❌ Failed to download dataset:", e)
        raise SystemExit("Dataset download failed — stopping pipeline.")
