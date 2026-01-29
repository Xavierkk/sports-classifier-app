import os
from supabase import create_client

# 1. CREDENTIALS (Found in Supabase Settings > API)
URL = "https://bsgcrzzizbgpafnvkkup.supabase.co"
# Use the 'service_role' secret key here so you have permission to upload
KEY = "sb_secret_mRbFF2jVxMGJnyICa8b1uA_VoJosNFJ" 
supabase = create_client(URL, KEY)

DATASET_PATH = "dataset" # The folder on your PC with train, test, valid
BUCKET_NAME = "sports-images"

def start_upload():
    print("🚀 Starting upload to Supabase...")
    
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Local path: dataset/train/cricket/1.jpg
                local_file = os.path.join(root, file)
                
                # Cloud path: train/cricket/1.jpg
                storage_path = os.path.relpath(local_file, DATASET_PATH).replace("\\", "/")
                
                try:
                    with open(local_file, "rb") as f:
                        # A. Upload to Storage
                        supabase.storage.from_(BUCKET_NAME).upload(
                            path=storage_path,
                            file=f,
                            file_options={"content-type": "image/jpeg", "x-upsert": "true"}
                        )
                    
                    # B. Get the Public URL
                    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)
                    
                    # C. Insert into your new PostgreSQL table
                    path_parts = storage_path.split('/')
                    supabase.table("image_metadata").insert({
                        "file_name": file,
                        "split": path_parts[0],       # 'train', 'test', or 'valid'
                        "sport_label": path_parts[1],  # The sport name folder
                        "image_url": public_url
                    }).execute()
                    
                    print(f"✅ Success: {storage_path}")
                    
                except Exception as e:
                    print(f"❌ Failed {storage_path}: {e}")

if __name__ == "__main__":
    start_upload()