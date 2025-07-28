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

    # 2. 获取图片生成提示词
    print("\n--- 步骤2: 获取图片生成提示词 ---")
    prompt = input("请输入图片生成提示词: ")
    if not prompt:
        print("提示词不能为空，流程终止。")
        return

    # 3. 发起图片生成请求
    print("\n--- 步骤3: 发起图片生成请求 ---")
    image_task_id = generate_image(prompt)
    if not image_task_id:
        print("图片生成请求失败，流程终止。")
        return

    # 4. 循环查询图片生成任务状态
    print(f"\n--- 步骤4: 查询图片生成任务状态 (任务ID: {image_task_id}) ---")
    image_url = None
    while image_url is None:
        print("等待图片生成完成...")
        time.sleep(10) # 每10秒查询一次
        image_url = query_image_generation_task(image_task_id)
        if image_url:
            print(f"图片生成成功，URL: {image_url}")
            break
        else:
            print("图片仍在生成中或查询失败，继续等待...")

    if not image_url:
        print("未能获取图片URL，流程终止。")
        return

    # 5. 发起视频生成请求
    print("\n--- 步骤5: 发起视频生成请求 ---")
    video_task_id = generate_video(image_url, prompt) # 传递图片生成的提示词
    if not video_task_id:
        print("视频生成请求失败，流程终止。")
        return

    # 6. 循环查询视频生成任务状态
    print(f"\n--- 步骤6: 查询视频生成任务状态 (任务ID: {video_task_id}) ---")
    video_url = None
    while video_url is None:
        print("等待视频生成完成...")
        time.sleep(10) # 每10秒查询一次
        video_url = query_image2video_task(video_task_id)
        if video_url:
            print(f"视频生成成功，URL: {video_url}")
            break
        else:
            print("视频仍在生成中或查询失败，继续等待...")

    if not video_url:
        print("未能获取视频URL，流程终止。")
        return

    print("\n--- 主流程执行完毕 ---")
    print(f"最终图片URL: {image_url}")
    print(f"最终视频URL: {video_url}")

if __name__ == "__main__":
    run_workflow()
