#!/usr/bin/env python3
"""
Image Renamer and Classifier using Gemini API
Analyzes images in the pics directory and renames them based on content description
Also provides functionality to classify and organize images into category folders
"""

import os
import re
import shutil
import json
import dashscope
from dashscope import MultiModalConversation
from PIL import Image
import argparse
from pathlib import Path

class ImageRenamer:
    def __init__(self, api_key=None, config_file="config.json"):
        """Initialize the ImageRenamer with DashScope API key"""
        if api_key:
            dashscope.api_key = api_key
        else:
            # Try to load API key from config file first
            api_key = self.load_api_key_from_config(config_file)
            if not api_key:
                # Try to get API key from environment variable
                api_key = os.getenv('DASHSCOPE_API_KEY')
                if not api_key:
                    raise ValueError("Please provide DashScope API key in config.json, as parameter, or set DASHSCOPE_API_KEY environment variable")
            dashscope.api_key = api_key
        
        # Initialize the model (Qwen-VL-Max for multimodal, which is the latest and most powerful version)
        self.model_name = 'qwen-vl-max'
        
        # Define common image categories
        self.categories = {
            'food': ['food', 'meal', 'cooking', 'restaurant', 'kitchen', 'drink', 'beverage', 'fruit', 'vegetable'],
            'people': ['person', 'people', 'human', 'child', 'adult', 'family', 'portrait', 'selfie'],
            'animals': ['animal', 'pet', 'dog', 'cat', 'bird', 'wildlife', 'zoo', 'farm'],
            'nature': ['landscape', 'mountain', 'forest', 'beach', 'ocean', 'sky', 'sunset', 'flower', 'tree'],
            'sports': ['sport', 'game', 'football', 'soccer', 'basketball', 'tennis', 'running', 'exercise'],
            'vehicles': ['car', 'truck', 'motorcycle', 'bicycle', 'plane', 'train', 'boat', 'vehicle'],
            'buildings': ['building', 'house', 'architecture', 'city', 'street', 'bridge', 'monument'],
            'art_design': ['art', 'painting', 'drawing', 'design', 'craft', 'handmade', 'creative', 'pattern'],
            'technology': ['computer', 'phone', 'device', 'electronic', 'gadget', 'screen', 'technology'],
            'characters': ['cartoon', 'anime', 'character', 'mascot', 'fictional', 'comic', 'animation']
        }
    
    def load_api_key_from_config(self, config_file):
        """Load API key from configuration file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('dashscope_api_key')
            return None
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {str(e)}")
            return None
    
    def sanitize_filename(self, text):
        """Convert text to a safe filename"""
        # Remove or replace invalid characters
        text = re.sub(r'[<>:"/\\|?*]', '', text)
        # Replace spaces and special chars with underscores
        text = re.sub(r'[\s\-\.]+', '_', text)
        # Remove multiple underscores
        text = re.sub(r'_+', '_', text)
        # Remove leading/trailing underscores
        text = text.strip('_')
        # Limit length
        text = text[:50]
        return text.lower()
    
    def analyze_image(self, image_path):
        """Analyze image using Qwen-VL-Plus and return description"""
        try:
            # Create prompt for image analysis
            prompt = """Analyze this image and provide a short, descriptive filename (2-4 words) that captures the main subject or content. 
            Focus on the most prominent elements like objects, people, animals, scenes, or activities.
            Respond with only the descriptive name, no additional text or explanation.
            Examples: "red_sports_car", "golden_retriever_dog", "sunset_beach_scene", "birthday_cake_candles"
            """
            
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'image': f'file://{image_path}'},
                        {'text': prompt}
                    ]
                }
            ]
            
            response = MultiModalConversation.call(model=self.model_name, messages=messages)
            
            if response.status_code == 200 and response.output.choices:
                # Extract text content from the response
                text_content = ""
                for item in response.output.choices[0].message.content:
                    if isinstance(item, dict) and 'text' in item:
                        text_content = item['text']
                        break
                if text_content:
                    return self.sanitize_filename(text_content.strip())
                else:
                    print(f"DashScope API: No text content found in response for analyze_image.")
                    return None
            else:
                print(f"DashScope API error for analyze_image: {response.message}")
                return None
                
        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return None
    
    def classify_image(self, image_path):
        """Classify image using Qwen-VL-Plus and return category"""
        try:
            # Create prompt for image classification
            prompt = """Analyze this image and classify it into one of these categories based on the main subject:
            
            Categories:
            - food: meals, cooking, drinks, fruits, vegetables, restaurants
            - people: humans, portraits, families, children, adults
            - animals: pets, wildlife, farm animals, zoo animals
            - nature: landscapes, mountains, forests, beaches, flowers, trees
            - sports: games, exercise, athletic activities, sports equipment
            - vehicles: cars, trucks, motorcycles, planes, trains, boats
            - buildings: architecture, houses, cities, streets, monuments
            - art_design: paintings, drawings, crafts, patterns, creative works
            - technology: computers, phones, electronics, gadgets, devices
            - characters: cartoons, anime, fictional characters, mascots
            
            Respond with only the category name (e.g., "food", "people", "nature"), no additional text.
            """
            
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'image': f'file://{image_path}'},
                        {'text': prompt}
                    ]
                }
            ]
            
            response = MultiModalConversation.call(model=self.model_name, messages=messages)
            
            if response.status_code == 200 and response.output.choices:
                # Extract text content from the response
                text_content = ""
                for item in response.output.choices[0].message.content:
                    if isinstance(item, dict) and 'text' in item:
                        text_content = item['text']
                        break
                
                if text_content:
                    category = text_content.strip().lower()
                    # Validate that the category is one of our predefined categories
                    if category in self.categories:
                        return category
                    else:
                        # Try to match based on keywords in the response
                        for cat, keywords in self.categories.items():
                            if any(keyword in category for keyword in keywords):
                                return cat
                        # Default to 'misc' if no match found
                        return 'misc'
                else:
                    print(f"DashScope API: No text content found in response for classify_image.")
                    return 'misc'
            else:
                print(f"DashScope API error for classify_image: {response.message}")
                return 'misc'
                
        except Exception as e:
            print(f"Error classifying image {image_path}: {str(e)}")
            return 'misc'
    
    def create_category_folder(self, base_directory, category):
        """Create category folder if it doesn't exist"""
        category_path = Path(base_directory) / category
        category_path.mkdir(exist_ok=True)
        return category_path
    
    def move_image_to_category(self, image_path, category, base_directory, dry_run=False):
        """Move image to appropriate category folder"""
        try:
            image_path = Path(image_path)
            category_folder = self.create_category_folder(base_directory, category)
            new_path = category_folder / image_path.name
            
            # Check if file already exists in destination
            counter = 1
            original_new_path = new_path
            while new_path.exists():
                stem = image_path.stem
                suffix = image_path.suffix
                new_path = category_folder / f"{stem}_{counter}{suffix}"
                counter += 1
            
            if dry_run:
                print(f"Would move: {image_path.name} -> {category}/{new_path.name}")
                return True
            else:
                # Move the file
                shutil.move(str(image_path), str(new_path))
                print(f"Moved: {image_path.name} -> {category}/{new_path.name}")
                return True
                
        except Exception as e:
            print(f"Error moving {image_path} to category {category}: {str(e)}")
            return False
    
    def rename_image(self, old_path, new_name):
        """Rename image file with new descriptive name"""
        try:
            old_path = Path(old_path)
            extension = old_path.suffix
            new_path = old_path.parent / f"{new_name}{extension}"
            
            # Check if new filename already exists
            counter = 1
            original_new_path = new_path
            while new_path.exists():
                new_path = old_path.parent / f"{new_name}_{counter}{extension}"
                counter += 1
            
            # Rename the file
            old_path.rename(new_path)
            print(f"Renamed: {old_path.name} -> {new_path.name}")
            return True
            
        except Exception as e:
            print(f"Error renaming {old_path}: {str(e)}")
            return False
    
    def process_directory(self, directory_path="pics", mode="rename", dry_run=False):
        """Process all images in the specified directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Directory {directory_path} does not exist!")
            return
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {directory_path}")
            return
        
        print(f"Found {len(image_files)} image(s) to process...")
        print(f"Mode: {mode}")
        
        for image_file in image_files:
            print(f"\nProcessing: {image_file.name}")
            
            if mode == "rename":
                # Analyze the image for renaming
                new_name = self.analyze_image(image_file)
                
                if new_name:
                    if dry_run:
                        print(f"Would rename: {image_file.name} -> {new_name}{image_file.suffix}")
                    else:
                        self.rename_image(image_file, new_name)
                else:
                    print(f"Failed to analyze image: {image_file.name}")
                    
            elif mode == "classify":
                # Classify the image and move to category folder
                category = self.classify_image(image_file)
                
                if category:
                    if dry_run:
                        print(f"Would classify as: {category}")
                    else:
                        print(f"Classified as: {category}")
                    self.move_image_to_category(image_file, category, directory_path, dry_run)
                else:
                    print(f"Failed to classify image: {image_file.name}")
                    
            elif mode == "both":
                # First rename, then classify
                new_name = self.analyze_image(image_file)
                category = self.classify_image(image_file)
                
                if new_name and category:
                    if dry_run:
                        print(f"Would rename: {image_file.name} -> {new_name}{image_file.suffix}")
                        print(f"Would classify as: {category}")
                        print(f"Final location: {category}/{new_name}{image_file.suffix}")
                    else:
                        # First rename the image
                        if self.rename_image(image_file, new_name):
                            # Update the path to the renamed file
                            renamed_path = image_file.parent / f"{new_name}{image_file.suffix}"
                            # Handle potential conflicts from renaming
                            counter = 1
                            while not renamed_path.exists() and counter < 10:
                                renamed_path = image_file.parent / f"{new_name}_{counter}{image_file.suffix}"
                                counter += 1
                            
                            if renamed_path.exists():
                                print(f"Classified as: {category}")
                                self.move_image_to_category(renamed_path, category, directory_path, dry_run)
                            else:
                                print(f"Could not find renamed file to classify")
                        else:
                            print(f"Failed to rename, skipping classification")
                else:
                    if not new_name:
                        print(f"Failed to analyze image: {image_file.name}")
                    if not category:
                        print(f"Failed to classify image: {image_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Rename and classify images using Gemini API analysis")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--directory", default="pics", help="Directory containing images (default: pics)")
    parser.add_argument("--mode", choices=["rename", "classify", "both"], default="rename", 
                       help="Processing mode: rename (rename files), classify (organize into folders), both (rename and classify)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    
    args = parser.parse_args()
    
    try:
        renamer = ImageRenamer(api_key=args.api_key)
        renamer.process_directory(args.directory, mode=args.mode, dry_run=args.dry_run)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo get a Gemini API key:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Set it as environment variable: export GEMINI_API_KEY='your-key-here'")
        print("4. Or pass it as argument: python image_renamer.py --api-key 'your-key-here'")
        print("\nUsage examples:")
        print("- Rename images: python image_renamer.py --mode rename")
        print("- Classify images: python image_renamer.py --mode classify")
        print("- Rename and classify: python image_renamer.py --mode both")
        print("- Dry run to see what would happen: python image_renamer.py --mode classify --dry-run")
    
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
