import subprocess
import time
from image_generation_request import generate_image
from image2video_request import generate_video
from query_image_generations import query_image_generation_task
from query_image2video_task import query_image2video_task

def run_workflow():
    print("--- 开始执行主流程 ---")

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

    # 5. 获取总生成次数
    print("\n--- 步骤5: 获取总生成次数 ---")
    try:
        total_generations = int(input("请输入总共要生成的次数: "))
        if total_generations <= 0:
            print("生成次数必须大于0，流程终止。")
            return
    except ValueError:
        print("无效的数字，流程终止。")
        return

    # 4. 获取提示词来源
    print("\n--- 步骤4: 获取提示词来源 ---")
    prompts = []
    prompt_file_path = input("请输入包含提示词的文件路径 (可选，留空则手动输入): ").strip()
    if prompt_file_path:
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            if not prompts:
                print("提示词文件为空或不包含有效提示词，将转为手动输入。")
                prompt_file_path = ""
            else:
                print(f"已从文件加载 {len(prompts)} 条提示词。")
                if len(prompts) < total_generations:
                    loop_choice = input(f"文件中的提示词数量 ({len(prompts)}) 少于总生成次数 ({total_generations})。\n是否循环使用文件中的提示词？(y/n): ").lower()
                    if loop_choice == 'y':
                        print("将循环使用文件中的提示词。")
                        prompt_loop_mode = True
                    else:
                        print("文件提示词用尽后，将提示手动输入。")
                        prompt_loop_mode = False
                else:
                    prompt_loop_mode = False # 不需要循环，因为提示词足够
        except FileNotFoundError:
            print("文件未找到，将转为手动输入。")
            prompt_file_path = ""
        except Exception as e:
            print(f"读取提示词文件时发生错误: {e}，将转为手动输入。")
            prompt_file_path = ""
    
    prompt_index = 0
    all_generated_image_urls = [] # 新增：用于存储所有生成的图片URL

    # 5. 循环执行生成流程
    print("\n--- 步骤5: 循环执行生成流程 ---")
    for i in range(total_generations):
        # 将 download_files 传递给查询函数
        print(f"\n--- 开始第 {i+1}/{total_generations} 次生成 ---")

        current_prompt = ""
        if prompt_file_path:
            if prompt_index < len(prompts):
                current_prompt = prompts[prompt_index]
                print(f"使用文件提示词: {current_prompt}")
                prompt_index += 1
            elif prompt_loop_mode:
                prompt_index = 0 # 循环使用，重置索引
                current_prompt = prompts[prompt_index]
                print(f"循环使用文件提示词: {current_prompt}")
                prompt_index += 1
            else:
                current_prompt = input("文件提示词已用尽，请输入新的图片生成提示词: ")
        else:
            current_prompt = input("请输入图片生成提示词: ")

        if not current_prompt:
            print("提示词不能为空，跳过当前生成。")
            continue

        # 5.1 发起图片生成请求
        print("\n--- 步骤5.1: 发起图片生成请求 ---")
        image_task_id = generate_image(current_prompt)
        if not image_task_id:
            print("图片生成请求失败，跳过当前生成。")
            continue

        # 5.2 循环查询图片生成任务状态
        print(f"\n--- 步骤5.2: 查询图片生成任务状态 (任务ID: {image_task_id}) ---")
        image_url = None
        start_time = time.time()
        while image_url is None and (time.time() - start_time < 300): # 设置5分钟超时
            print("等待图片生成完成...")
            time.sleep(10) # 每10秒查询一次
            image_url = query_image_generation_task(image_task_id, download_images) # 传递 download_images 参数
            if image_url:
                print(f"图片生成成功，URL: {image_url}")
                all_generated_image_urls.append(image_url) # 存储图片URL
                break
            else:
                print("图片仍在生成中或查询失败，继续等待...")

        if not image_url:
            print("未能获取图片URL，跳过当前生成。")
            continue

        if generate_video_mode:
            # 5.3 发起视频生成请求
            print("\n--- 步骤5.3: 发起视频生成请求 ---")
            video_task_id = generate_video(image_url, current_prompt)
            if not video_task_id:
                print("视频生成请求失败，跳过当前视频生成。")
                continue

            # 5.4 循环查询视频生成任务状态
            print(f"\n--- 步骤5.4: 查询视频生成任务状态 (任务ID: {video_task_id}) ---")
            video_url = None
            start_time = time.time()
            while video_url is None and (time.time() - start_time < 300): # 设置5分钟超时
                print("等待视频生成完成...")
                time.sleep(10) # 每10秒查询一次
                video_url = query_image2video_task(video_task_id, download_videos) # 传递 download_videos 参数
                if video_url:
                    print(f"视频生成成功，URL: {video_url}")
                    break
                else:
                    print("视频仍在生成中或查询失败，继续等待...")

            if not video_url:
                print("未能获取视频URL，跳过当前视频生成。")
                continue
            
            print(f"最终图片URL: {image_url}")
            print(f"最终视频URL: {video_url}")
        else:
            print(f"仅图片模式，最终图片URL: {image_url}")

    print("\n--- 主流程执行完毕 ---")
    # 汇总输出所有生成的图片URL
    if all_generated_image_urls:
        print("\n--- 所有生成的图片URL汇总 ---")
        for idx, url in enumerate(all_generated_image_urls):
            print(f"图片 {idx+1}: {url}")
    else:
        print("\n没有成功生成任何图片URL。")

if __name__ == "__main__":
    run_workflow()
