import requests
import json
import os
import urllib.request

def download_file(url: str, save_path: str):
    """下载文件到指定路径"""
    try:
        print(f"正在下载文件: {url} 到 {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"文件下载成功: {save_path}")
        return True
    except Exception as e:
        print(f"文件下载失败: {url}, 错误: {e}")
        return False

def query_image2video_task(task_id: str, download_files: bool = False):
    print(f"开始查询视频生成任务: {task_id}...")

    # 从api_token.txt读取鉴权信息
    with open('api_token.txt', 'r') as f:
        api_token = f.read().strip()

    # API请求的URL
    url = f"https://api-beijing.klingai.com/v1/videos/image2video/{task_id}" # 查询特定任务的URL

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    # 发送GET请求
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status() # 检查HTTP请求是否成功

        print(f"查询视频生成任务 {task_id} 成功！")
        response_data = response.json()
        print("响应内容:", json.dumps(response_data, indent=4, ensure_ascii=False))

        if response_data and response_data.get("code") == 0 and response_data.get("data"):
            task_status = response_data["data"].get("task_status")
            print(f"任务状态: {task_status}")
            if task_status == "succeed":
                task_result = response_data["data"].get("task_result")
                if task_result and task_result.get("videos"):
                    video_url = task_result["videos"][0].get("url") # 假设只生成一个视频
                    if video_url:
                        print(f"视频URL: {video_url}")
                        
                        if download_files:
                            # 确保下载目录存在
                            download_dir = "downloads"
                            os.makedirs(download_dir, exist_ok=True)
                            
                            # 从URL中提取文件名，并去除查询参数
                            file_name = os.path.basename(video_url)
                            if '?' in file_name:
                                file_name = file_name.split('?')[0]
                            save_path = os.path.join(download_dir, file_name)
                            download_file(video_url, save_path)

                        return video_url
            return None
        return None

    except requests.exceptions.RequestException as e:
        print(f"查询视频生成任务 {task_id} 失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None
