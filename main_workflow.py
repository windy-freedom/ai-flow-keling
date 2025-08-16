import subprocess
import time
import os
import json
import datetime # New import for datetime.date.today()
import dashscope
from dashscope import Generation
from dashscope import MultiModalConversation # New import for image classification
from image_generation_request import generate_image
from image2video_request import generate_video
from query_image_generations import query_image_generation_task
from query_image2video_task import query_image2video_task
from image_renamer import ImageRenamer # New import for image classification
from pathlib import Path # New import for file path handling
import shutil # New import for file operations
import re # New import for regex in classification

def generate_prompts(keyword: str, num_prompts: int):
    """
    Generates image/video prompts using the Qwen model via DashScope.
    """
    # Construct the prompt for the Qwen model
    # The prompt asks for a specific number of prompts, each on a new line.
    qwen_prompt = f"请为关键词'{keyword}'生成{num_prompts}个用于图片和视频的提示词，丰富一点，每条提示词换行。"

    print(f"Sending prompt to Qwen model: '{qwen_prompt}'")

    try:
        # Call the Qwen model for text generation
        # Using 'qwen-turbo' as a general text generation model.
        response = Generation.call(
            model='qwen-turbo',
            prompt=qwen_prompt
        )

        if response.status_code == 200 and response.output.text:
            generated_text = response.output.text.strip()
            print("Generated text from Qwen model:")
            print(generated_text)
            
            # Split the generated text into individual prompts
            # Assuming each prompt is on a new line as requested in the qwen_prompt
            prompts = [p.strip() for p in generated_text.split('\n') if p.strip()]
            return prompts
        else:
            print(f"DashScope API error: {response.message}")
            return []
    except Exception as e:
        print(f"Error calling DashScope API: {e}")
        return []

def save_prompts_to_file(prompts: list, keyword: str):
    """
    Saves the generated prompts to a new file named by date and keyword.
    """
    today_date = datetime.date.today().strftime("%Y-%m-%d")
    filename = f"{today_date}-{keyword}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
        print(f"Prompts saved to {filename}")
    except Exception as e:
        print(f"Error saving prompts to file {filename}: {e}")

def image_to_prompt(image_path: Path):
    """
    Generates a descriptive text prompt from an image using a multimodal model.
    """
    print(f"--- 开始从图片生成提示词: {image_path.name} ---")
    try:
        prompt_text = """请详细描述这张图片的内容，包括主体、背景、风格、色彩、光线、构图等，生成一个用于图片或视频生成的详细提示词。
        请直接给出提示词，不要包含任何额外说明或前缀。
        """
        
        messages = [
            {
                'role': 'user',
                'content': [
                    {'image': f'file://{image_path}'},
                    {'text': prompt_text}
                ]
            }
        ]
        
        response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
        
        if response.status_code == 200 and response.output.choices:
            generated_prompt = ""
            for item in response.output.choices[0].message.content:
                if isinstance(item, dict) and 'text' in item:
                    generated_prompt = item['text'].strip()
                    break
            
            if generated_prompt:
                print(f"  → 生成的提示词: {generated_prompt}")
                print(f"--- 图片到提示词生成完成 ---")
                return generated_prompt
            else:
                print(f"DashScope API: 未能从图片生成有效提示词。")
                return None
        else:
            print(f"DashScope API error for image_to_prompt: {response.message}")
            return None
        
    except Exception as e:
        print(f"从图片生成提示词失败: {e}")
        return None

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

def classify_single_image(image_path: Path):
    """Classifies and organizes a single image based on AI analysis."""
    print(f"\n--- 开始对图片进行智能分类和重命名: {image_path.name} ---")
    try:
        renamer = ImageRenamer()
        
        original_image_path = image_path # Keep original path for reference
        
        # 1. Rename the image
        new_name = renamer.analyze_image(original_image_path)
        if new_name:
            # Construct the new path with the descriptive name
            new_image_path_after_rename = original_image_path.parent / f"{new_name}{original_image_path.suffix}"
            
            # Handle potential filename conflicts during renaming
            counter = 1
            temp_new_image_path = new_image_path_after_rename
            while temp_new_image_path.exists() and temp_new_image_path != original_image_path:
                temp_new_image_path = original_image_path.parent / f"{new_name}_{counter}{original_image_path.suffix}"
                counter += 1
            new_image_path_after_rename = temp_new_image_path

            try:
                os.rename(str(original_image_path), str(new_image_path_after_rename))
                print(f"  → 重命名为: {new_image_path_after_rename.name}")
                image_path = new_image_path_after_rename # Update image_path to the new name
            except Exception as e:
                print(f"  ✗ 重命名失败 {original_image_path.name} 到 {new_image_path_after_rename.name}: {e}")
                # If rename fails, continue with original name for classification
                image_path = original_image_path
        else:
            print(f"  → 未能生成描述性名称，保留原文件名: {image_path.name}")

        # 2. Classify the (potentially renamed) image
        category = get_smart_category(renamer, image_path)
        print(f"  → AI建议分类: {category}")
        
        pics_dir = Path("downloads")
        category_dir = pics_dir / category
        category_dir.mkdir(exist_ok=True)
        
        final_filename = image_path.name
        destination = category_dir / final_filename
        
        # Handle filename conflicts when moving to category folder
        counter = 1
        while destination.exists():
            stem = image_path.stem
            suffix = image_path.suffix
            destination = category_dir / f"{stem}_{counter}{suffix}"
            counter += 1
            
        shutil.move(str(image_path), str(destination))
        print(f"  → 移动到: {category}/{destination.name}")
        print(f"--- 图片分类和重命名完成 ---")
        
    except Exception as e:
        print(f"图片分类或重命名失败: {e}")

def evaluate_and_select_best_image(image_urls: list, original_prompt: str):
    """
    Evaluates a list of image URLs based on a given prompt using AI,
    and selects the best one based on matching, aesthetics, and clarity.
    Returns the URL of the best image.
    """
    print("\n--- 开始AI评价并选择最佳图片 ---")
    best_image_url = None
    highest_score = -1 # Assuming a score from 0-100 or similar

    evaluation_prompt_template = f"""请评价这张图片与以下提示词的匹配度、美观度和清晰度。
    原始提示词: "{original_prompt}"

    请根据以下标准给出1到100的综合评分，并简要说明理由。
    评分标准：
    1.  **匹配度 (Prompt Adherence):** 图片内容与原始提示词的符合程度。
    2.  **美观度 (Aesthetics):** 图片的整体视觉吸引力、色彩和谐、构图平衡等。
    3.  **清晰度 (Clarity):** 图片的细节、锐利度、是否有模糊或噪点。

    请以JSON格式返回结果，包含'score' (整数) 和 'reason' (字符串)。
    例如: {{"score": 85, "reason": "图片与提示词高度匹配，色彩鲜明，构图良好，但略有模糊。"}}
    """

    for idx, image_url in enumerate(image_urls):
        print(f"  → 评价图片 {idx+1}/{len(image_urls)}: {image_url}")
        try:
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'image': image_url}, # Use image URL directly
                        {'text': evaluation_prompt_template}
                    ]
                }
            ]
            
            response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages) # Use qwen-vl-plus for evaluation
            
            if response.status_code == 200 and response.output.choices:
                evaluation_result_text = ""
                for item in response.output.choices[0].message.content:
                    if isinstance(item, dict) and 'text' in item:
                        evaluation_result_text = item['text'].strip()
                        break
                
                try:
                    # Attempt to parse the JSON response
                    evaluation_data = json.loads(evaluation_result_text)
                    score = evaluation_data.get('score', 0)
                    reason = evaluation_data.get('reason', '无理由')
                    
                    print(f"    评分: {score}, 理由: {reason}")

                    if score > highest_score:
                        highest_score = score
                        best_image_url = image_url
                        print(f"    当前最佳图片更新为: {best_image_url} (得分: {highest_score})")
                except json.JSONDecodeError:
                    print(f"    AI返回结果非JSON格式，尝试从文本中提取分数: {evaluation_result_text}")
                    # Fallback: try to extract score from text if JSON parsing fails
                    score_match = re.search(r'"score":\s*(\d+)', evaluation_result_text)
                    if score_match:
                        score = int(score_match.group(1))
                        print(f"    从文本中提取到分数: {score}")
                        if score > highest_score:
                            highest_score = score
                            best_image_url = image_url
                            print(f"    当前最佳图片更新为: {best_image_url} (得分: {highest_score})")
                    else:
                        print(f"    未能从AI评价中提取有效分数。")
            else:
                print(f"DashScope API error for image evaluation: {response.message}")
        except Exception as e:
            print(f"评价图片 {image_url} 时发生错误: {e}")
            
    if best_image_url:
        print(f"\n--- AI评价完成，选出最佳图片: {best_image_url} (最高得分: {highest_score}) ---")
    else:
        print("\n--- 未能选出最佳图片。---")
    return best_image_url

def run_workflow():
    print("--- 开始执行主流程 ---")

    # Load DashScope API key
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            dashscope.api_key = config.get('dashscope_api_key')
            if not dashscope.api_key:
                print("Error: 'dashscope_api_key' not found in config.json or is empty.")
                return
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}. Please ensure it exists.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode config.json. Please check its format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading config.json: {e}")
        return

    # 1. 生成并保存API Token
    print("\n--- 步骤1: 生成API Token ---")
    try:
        subprocess.run(["python", "generate_token.py"], check=True)
        print("API Token 生成并保存成功。")
    except subprocess.CalledProcessError as e:
        print(f"API Token 生成失败: {e}")
        return

    # 2. 选择生成模式
    print("\n--- 步骤2: 选择生成模式 ---")
    mode_choice = input("请选择生成模式 (1: 循环生成图片和视频, 2: 仅循环生成图片): ")
    if mode_choice not in ['1', '2']:
        print("无效的选择，流程终止。")
        return
    generate_video_mode = (mode_choice == '1')

    # 3. 获取图片下载选项
    print("\n--- 步骤3: 获取图片下载选项 ---")
    download_image_choice = input("是否下载生成的图片到本地？(y/n): ").lower()
    download_images = (download_image_choice == 'y')
    if not download_images:
        print("将不会下载生成的图片到本地。")

    # 4. 获取视频下载选项
    print("\n--- 步骤4: 获取视频下载选项 ---")
    download_video_choice = input("是否下载生成的视频到本地？(y/n): ").lower()
    download_videos = (download_video_choice == 'y')
    if not download_videos:
        print("将不会下载生成的视频到本地。")

    # 5. 获取总生成次数 和 提示词来源
    print("\n--- 步骤5: 获取总生成次数和提示词来源 ---")
    prompt_source_choice = input("请选择提示词来源 (1: 从图片生成, 2: 输入关键词): ")
    initial_prompt = ""
    keyword = "" # Initialize keyword
    if prompt_source_choice == '1':
        image_path_str = input("请输入图片文件路径 (e.g., downloads/my_image.png): ")
        image_file_path = Path(image_path_str)
        if not image_file_path.exists():
            print(f"错误: 图片文件 '{image_file_path}' 不存在。流程终止。")
            return
        initial_prompt = image_to_prompt(image_file_path)
        if not initial_prompt:
            print("未能从图片生成提示词，流程终止。")
            return
        print(f"已从图片生成初始提示词: {initial_prompt}")
        keyword = initial_prompt # Use the generated prompt as the keyword for saving
    elif prompt_source_choice == '2':
        keyword = input("请输入关键词 (e.g., 猫): ")
        initial_prompt = keyword # For keyword input, the initial prompt is the keyword itself
    else:
        print("无效的选择，流程终止。")
        return

    while True:
        try:
            num_images_to_generate_str = input("请输入要生成的图片数量 (建议3-5张用于AI评价): ")
            num_images_to_generate = int(num_images_to_generate_str)
            if num_images_to_generate <= 0:
                print("图片数量必须是正整数。")
            else:
                break
        except ValueError:
            print("无效的输入。请输入一个整数。")

    # For now, we'll generate prompts based on the initial_prompt/keyword
    # Later, we'll use the initial_prompt directly for image generation
    # and generate auxiliary prompts for video.
    prompts = [initial_prompt] * num_images_to_generate # Generate multiple images from the same initial prompt
    total_generations = num_images_to_generate # Total generations will be the number of images to generate for evaluation
    
    if not prompts:
        print("未能生成提示词，流程终止。")
        return
    else:
        print(f"将生成 {len(prompts)} 张图片用于AI评价。")
        save_prompts_to_file(prompts, keyword) # Save generated prompts to a file

    # Initialize all_generated_image_urls here, before the loop
    all_generated_image_urls = [] 
    
    # 6. 循环执行生成流程 (原步骤5)
    print("\n--- 步骤6: 循环执行生成流程 ---")
    # The loop will now iterate 'num_images_to_generate' times, each time using the 'initial_prompt'
    for i in range(total_generations): # total_generations is now num_images_to_generate
        print(f"\n--- 开始第 {i+1}/{total_generations} 次图片生成 ---")

        current_prompt = initial_prompt # Always use the initial_prompt for image generation

        if not current_prompt:
            print("提示词不能为空，跳过当前生成。")
            continue

        # 6.1 发起图片生成请求 (原步骤5.1)
        print("\n--- 步骤6.1: 发起图片生成请求 ---")
        image_task_id = generate_image(current_prompt)
        if not image_task_id:
            print("图片生成请求失败，跳过当前生成。")
            continue

        # 6.2 循环查询图片生成任务状态 (原步骤5.2)
        print(f"\n--- 步骤6.2: 查询图片生成任务状态 (任务ID: {image_task_id}) ---")
        image_url = None
        start_time = time.time()
        while image_url is None and (time.time() - start_time < 300): # 设置5分钟超时
            print("等待图片生成完成...")
            time.sleep(10) # 每10秒查询一次
            image_url = query_image_generation_task(image_task_id, download_images)
            if image_url:
                print(f"图片生成成功，URL: {image_url}")
                all_generated_image_urls.append(image_url)
                break
            else:
                print("图片仍在生成中或查询失败，继续等待...")

        if not image_url:
            print("未能获取图片URL，跳过当前生成。")
            continue

    # After generating all images, proceed with evaluation and video generation
    if not all_generated_image_urls:
        print("\n没有成功生成任何图片，流程终止。")
        return

    # 7. AI评价并选择最佳图片
    print("\n--- 步骤7: AI评价并选择最佳图片 ---")
    best_image_url = evaluate_and_select_best_image(all_generated_image_urls, initial_prompt)

    if not best_image_url:
        print("未能选出最佳图片，流程终止。")
        return

    print(f"\n--- 最佳图片URL: {best_image_url} ---")

    if generate_video_mode:
        # 8. 为最佳图片生成辅助提示词
        print("\n--- 步骤8: 为最佳图片生成辅助提示词 ---")
        # Use the initial_prompt as a base for generating auxiliary prompt
        auxiliary_prompts = generate_prompts(initial_prompt, 1) # Generate 1 auxiliary prompt
        auxiliary_prompt = auxiliary_prompts[0] if auxiliary_prompts else initial_prompt # Fallback to initial_prompt
        print(f"  → 生成的辅助提示词: {auxiliary_prompt}")

        # 9. 发起视频生成请求
        print("\n--- 步骤9: 发起视频生成请求 ---")
        video_task_id = generate_video(best_image_url, auxiliary_prompt)
        if not video_task_id:
            print("视频生成请求失败，流程终止。")
            return

        # 10. 循环查询视频生成任务状态
        print(f"\n--- 步骤10: 查询视频生成任务状态 (任务ID: {video_task_id}) ---")
        video_url = None
        start_time = time.time()
        while video_url is None and (time.time() - start_time < 300): # 设置5分钟超时
            print("等待视频生成完成...")
            time.sleep(10) # 每10秒查询一次
            video_url = query_image2video_task(video_task_id, download_videos)
            if video_url:
                print(f"视频生成成功，URL: {video_url}")
                break
            else:
                print("视频仍在生成中或查询失败，继续等待...")

        if not video_url:
            print("未能获取视频URL，流程终止。")
            return
        
        print(f"\n--- 最终流程结果 ---")
        print(f"最佳图片URL: {best_image_url}")
        print(f"生成的视频URL: {video_url}")
    else:
        print("\n--- 仅图片模式，流程结束 ---")
        print(f"最佳图片URL: {best_image_url}")
        # If in image-only mode and image was downloaded, classify it
        if download_images:
            # Reconstruct the local path of the downloaded image
            download_dir = "downloads"
            file_name = os.path.basename(best_image_url)
            if '?' in file_name:
                file_name = file_name.split('?')[0]
            local_image_path = Path(download_dir) / file_name
            
            if local_image_path.exists():
                classify_single_image(local_image_path)
            else:
                print(f"警告: 未找到下载的图片文件 {local_image_path}，跳过分类。")

    print("\n--- 主流程执行完毕 ---")

if __name__ == "__main__":
    run_workflow()
