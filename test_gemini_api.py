import os
import google.generativeai as genai
from PIL import Image
from pathlib import Path
import json

def load_api_key_from_config(config_file="config.json"):
    """Load API key from configuration file"""
    try:
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('gemini_api_key')
        return None
    except Exception as e:
        print(f"Warning: Could not load config file {config_file}: {str(e)}")
        return None

def test_api_call():
    try:
        api_key = load_api_key_from_config()
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("Error: GEMINI_API_KEY not found in config.json or environment variable.")
                return

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Use a dummy image or a small image from downloads for testing
        # For simplicity, let's try to generate some text without an image first
        print("Attempting a simple text generation with Gemini API...")
        response = model.generate_content("Hello, Gemini!")
        print(f"API Response (text): {response.text}")
        
        # Now try with an image if available
        downloads_dir = Path("downloads")
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        for ext in image_extensions:
            image_files.extend(downloads_dir.glob(f"*{ext}"))
            image_files.extend(downloads_dir.glob(f"*{ext.upper()}"))

        if image_files:
            test_image_path = image_files[0]
            print(f"\nAttempting image analysis with Gemini API using: {test_image_path}")
            image = Image.open(test_image_path)
            prompt = "Describe this image in one sentence."
            response = model.generate_content([prompt, image])
            print(f"API Response (image description): {response.text}")
        else:
            print("\nNo image files found in 'downloads' directory for image analysis test.")

    except Exception as e:
        print(f"An error occurred during API test: {e}")

if __name__ == "__main__":
    test_api_call()
