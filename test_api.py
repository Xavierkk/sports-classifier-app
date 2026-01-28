import requests
import os
import random

# 1. Configuration
URL = "https://sports-classifier-app-xavier.onrender.com/predict"
# Since the folder is in the same directory, we use a relative path
BASE_PATH = os.path.join(os.getcwd(), "dataset", "test")

def test_random_images(num_tests=5):
    # Check if path exists
    if not os.path.exists(BASE_PATH):
        print(f"❌ Error: Cannot find folder at {BASE_PATH}")
        return

    # Get list of all sport folders
    sport_folders = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]
    
    print(f"🚀 Starting Test on {URL}")
    print(f"Checking {num_tests} random images from {len(sport_folders)} categories...\n")

    for i in range(num_tests):
        # Pick a random sport and image
        actual_sport = random.choice(sport_folders)
        sport_dir = os.path.join(BASE_PATH, actual_sport)
        image_name = random.choice(os.listdir(sport_dir))
        image_path = os.path.join(sport_dir, image_name)

        try:
            with open(image_path, "rb") as img:
                files = {"file": (image_name, img, "image/jpeg")}
                # Timeout is 60s in case Render is "waking up"
                response = requests.post(URL, files=files, timeout=60)

            if response.status_code == 200:
                result = response.json()
                predicted = result.get('sport')
                version = result.get('version', 'N/A')
                
                # Check if correct
                match = "✅" if predicted.lower() == actual_sport.lower() else "❌"
                
                print(f"[{i+1}] {match} Actual: {actual_sport.ljust(15)} | Predicted: {predicted.ljust(15)} | API: {version}")
            else:
                print(f"[{i+1}] ❌ API Error: {response.status_code}")

        except Exception as e:
            print(f"[{i+1}] 💥 Error: {e}")

if __name__ == "__main__":
    test_random_images(5)