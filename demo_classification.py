#!/usr/bin/env python3
"""
Demo script to intelligently classify and organize all images based on AI analysis
"""

import os
import shutil
import re
from pathlib import Path
from image_renamer import ImageRenamer
from dashscope import MultiModalConversation, Generation

def get_smart_category(renamer, image_path):
    """Use AI to intelligently determine the best category for an image"""
    try:
        prompt = """Analyze this image and determine the most appropriate category folder name for organizing it.
        
        Look at the main subject/content and suggest a simple, descriptive category name (1-2 words, lowercase, use underscore for spaces).
        
        Examples of good category names:
        - cats, dogs, birds, animals
        - people, children, portraits
        - food, cooking, drinks
        - nature, landscapes, flowers
        - cars, vehicles, transportation
        - buildings, architecture
        - art, paintings, drawings
        - technology, computers, phones
        - sports, games, activities
        
        Respond with only the category name, no additional text or explanation.
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
        
        response = MultiModalConversation.call(model=renamer.model_name, messages=messages)
        
        if response.status_code == 200 and response.output.choices:
            # Extract text content from the response
            text_content = ""
            for item in response.output.choices[0].message.content:
                if isinstance(item, dict) and 'text' in item:
                    text_content = item['text']
                    break
            
            if text_content:
                category = text_content.strip().lower()
                # Clean up the category name
                category = re.sub(r'[^a-z0-9_]', '', category)
                category = re.sub(r'_+', '_', category)
                category = category.strip('_')
                
                if category and len(category) > 0:
                    return category
                else:
                    return 'misc'
            else:
                print(f"DashScope API: No text content found in response for get_smart_category.")
                return 'misc'
        else:
            print(f"DashScope API error for get_smart_category: {response.message}")
            return 'misc'
        
    except Exception as e:
        print(f"Error getting smart category for {image_path}: {e}")
        return 'misc'

def read_file_content(file_path):
    """Reads the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def analyze_text_content(text_content, api_key=None):
    """Analyzes text content using Qwen-turbo and returns a descriptive name."""
    if not text_content:
        return None
    
    prompt = """Analyze the following text content and provide a short, descriptive filename (2-4 words) that captures its main subject or content.
    Focus on the most prominent elements like topics, themes, or key entities.
    Respond with only the descriptive name, no additional text or explanation.
    Examples: "project_report", "meeting_minutes", "travel_guide", "recipe_book"
    
    Text content:
    """ + text_content[:1000] # Limit text content to avoid exceeding token limits

    try:
        response = Generation.call(
            model='qwen-turbo',
            prompt=prompt,
            api_key=api_key # Use provided API key if available
        )
        
        if response.status_code == 200 and response.output.text:
            # Sanitize filename similar to ImageRenamer
            text_name = response.output.text.strip()
            text_name = re.sub(r'[<>:"/\\|?*]', '', text_name)
            text_name = re.sub(r'[\s\-\.]+', '_', text_name)
            text_name = re.sub(r'_+', '_', text_name)
            text_name = text_name.strip('_')
            text_name = text_name[:50] # Limit length
            return text_name.lower()
        else:
            print(f"DashScope API error for analyze_text_content: {response.message}")
            return None
    except Exception as e:
        print(f"Error analyzing text content: {e}")
        return None

def classify_text_content(text_content, api_key=None):
    """Classifies text content using Qwen-turbo and returns a category."""
    if not text_content:
        return 'misc'
    
    prompt = """Analyze the following text content and classify it into one of these general categories:
    
    Categories:
    - documents: reports, articles, essays, research papers, official records
    - notes: meeting minutes, personal notes, memos, drafts
    - code: programming scripts, configuration files, logs, technical specifications
    - creative: stories, poems, scripts, lyrics, artistic descriptions
    - data: lists, tables, raw data, spreadsheets, databases
    - communication: emails, chats, messages, letters, transcripts
    - misc: anything that doesn't fit well into other categories
    
    Respond with only the category name (e.g., "documents", "notes", "code"), no additional text.
    
    Text content:
    """ + text_content[:1000] # Limit text content

    try:
        response = Generation.call(
            model='qwen-turbo',
            prompt=prompt,
            api_key=api_key # Use provided API key if available
        )
        
        if response.status_code == 200 and response.output.text:
            category = response.output.text.strip().lower()
            category = re.sub(r'[^a-z0-9_]', '', category)
            category = re.sub(r'_+', '_', category)
            category = category.strip('_')
            
            # Validate against a predefined list of categories or default to 'misc'
            valid_categories = {'documents', 'notes', 'code', 'creative', 'data', 'communication', 'misc'}
            if category in valid_categories:
                return category
            else:
                return 'misc'
        else:
            print(f"DashScope API error for classify_text_content: {response.message}")
            return 'misc'
    except Exception as e:
        print(f"Error classifying text content: {e}")
        return 'misc'

def classify_and_organize_all_media(rename_files=True):
    """Intelligently classify and organize all media (images and text) based on AI analysis, with optional renaming"""
    print("=== Smart Media Classification and Organization ===\n")
    
    try:
        # Initialize the renamer (it will read API key from config.json)
        renamer = ImageRenamer()
        
        # Define the directory containing media files
        media_dir = Path("downloads")
        
        # Supported media extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        text_extensions = {'.txt', '.md'} # Add more as needed
        
        # Find all media files in the specified directory (not in subdirectories)
        media_files = []
        for ext in image_extensions:
            media_files.extend(media_dir.glob(f"*{ext}"))
            media_files.extend(media_dir.glob(f"*{ext.upper()}"))
        for ext in text_extensions:
            media_files.extend(media_dir.glob(f"*{ext}"))
            media_files.extend(media_dir.glob(f"*{ext.upper()}"))
        
        # Convert to a list to avoid issues with files being moved during iteration
        media_files = list(media_files)
        
        if not media_files:
            print(f"No media files found in the specified directory to classify")
            return
        
        print(f"Found {len(media_files)} media file(s) to analyze and organize...")
        if rename_files:
            print("Mode: Rename + Smart Classification")
        else:
            print("Mode: Smart Classification Only")
        
        # Track categories and counts
        category_counts = {}
        created_folders = set()
        
        for media_file in media_files:
            print(f"\nAnalyzing: {media_file.name}")
            
            file_extension = media_file.suffix.lower()
            
            new_name = None
            category = None
            
            if file_extension in image_extensions:
                # Process as image
                category = get_smart_category(renamer, media_file)
                print(f"  → AI suggested category (Image): {category}")
                if rename_files:
                    new_name = renamer.analyze_image(media_file)
                    if new_name:
                        print(f"  → AI suggested name (Image): {new_name}")
                    else:
                        print(f"  → Could not generate new name for image, keeping original")
                        new_name = media_file.stem
            elif file_extension in text_extensions:
                # Process as text
                text_content = read_file_content(media_file)
                if text_content:
                    category = classify_text_content(text_content, api_key=renamer.api_key) # Pass API key
                    print(f"  → AI suggested category (Text): {category}")
                    if rename_files:
                        new_name = analyze_text_content(text_content, api_key=renamer.api_key) # Pass API key
                        if new_name:
                            print(f"  → AI suggested name (Text): {new_name}")
                        else:
                            print(f"  → Could not generate new name for text, keeping original")
                            new_name = media_file.stem
                else:
                    print(f"  → Could not read text file content: {media_file.name}")
                    new_name = media_file.stem # Keep original name if content can't be read
                    category = 'misc' # Default category
            else:
                print(f"  → Unsupported file type: {media_file.name}, skipping.")
                continue # Skip unsupported files
            
            if not category: # Fallback if AI classification fails for any reason
                category = 'misc'
            if not new_name: # Fallback if AI renaming fails for any reason
                new_name = media_file.stem

            # Create category folder if it doesn't exist
            category_dir = media_dir / category
            if category not in created_folders:
                category_dir.mkdir(exist_ok=True)
                created_folders.add(category)
                if category not in category_counts:
                    category_counts[category] = 0
            
            # Determine final filename
            final_filename = f"{new_name}{media_file.suffix}"
            
            # Move to category folder
            destination = category_dir / final_filename
            
            # Handle filename conflicts
            counter = 1
            while destination.exists():
                stem = new_name # Use the new_name as stem for conflict resolution
                suffix = media_file.suffix
                destination = category_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            try:
                shutil.move(str(media_file), str(destination))
                if rename_files and new_name and destination.name != media_file.name:
                    print(f"  → Renamed and moved to: {category}/{destination.name}")
                else:
                    print(f"  → Moved to: {category}/{destination.name}")
                category_counts[category] += 1
            except Exception as e:
                print(f"  ✗ Error moving file: {e}")
        
        print(f"\n=== Classification Summary ===")
        print(f"Total media processed: {len(media_files)}")
        print(f"Categories created: {len(created_folders)}")
        
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} media files")
        
        # Show contents of each category folder
        print(f"\n=== Folder Contents ===")
        for category in sorted(created_folders):
            category_path = media_dir / category
            files = [f for f in category_path.glob("*") if f.is_file() and f.suffix.lower() in (image_extensions | text_extensions)]
            if files:
                print(f"\n{category}/ ({len(files)} files):")
                for file in sorted(files):
                    print(f"  - {file.name}")
        
    except Exception as e:
        print(f"Error: {e}")

def identify_and_organize_animals():
    """Legacy function - now calls the smart classification function"""
    classify_and_organize_all_media()

def demo_classification():
    """Original demo function for backward compatibility"""
    print("=== Image Classification Demo ===\n")
    
    try:
        # Initialize the renamer
        renamer = ImageRenamer()
        
        print("Available categories:")
        for category, keywords in renamer.categories.items():
            print(f"  {category}: {', '.join(keywords[:5])}...")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function with mode selection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Image Classification and Organization with AI (using Qwen-VL-Plus)")
    parser.add_argument("--mode", choices=["classify", "rename-classify", "classify-only"], 
                       default="rename-classify",
                       help="Processing mode: classify (legacy), rename-classify (rename + smart classify), classify-only (smart classify without renaming)")
    parser.add_argument("--directory", default="downloads", help="Directory containing images (default: downloads)")
    
    args = parser.parse_args()
    
    if args.mode == "classify":
        # Legacy mode using original classification
        identify_and_organize_animals()
    elif args.mode == "rename-classify":
        # New mode: rename + smart classification
        classify_and_organize_all_media(rename_files=True)
    elif args.mode == "classify-only":
        # New mode: smart classification without renaming
        classify_and_organize_all_media(rename_files=False)

if __name__ == "__main__":
    main()
