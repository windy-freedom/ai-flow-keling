import requests
import json

def query_image_generation_task(task_id: str):
    print(f"开始查询图片生成任务: {task_id}...")

    # 从api_token.txt读取鉴权信息
    with open('api_token.txt', 'r') as f:
        api_token = f.read().strip()

    # API请求的URL
    url = f"https://api-beijing.klingai.com/v1/images/generations/{task_id}" # 查询特定任务的URL

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    # 发送GET请求
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status() # 检查HTTP请求是否成功

        print(f"查询图片生成任务 {task_id} 成功！")
        response_data = response.json()
        print("响应内容:", json.dumps(response_data, indent=4, ensure_ascii=False))

        if response_data and response_data.get("code") == 0 and response_data.get("data"):
            task_status = response_data["data"].get("task_status")
            print(f"任务状态: {task_status}")
            if task_status == "succeed":
                task_result = response_data["data"].get("task_result")
                if task_result and task_result.get("images"):
                    image_url = task_result["images"][0].get("url") # 假设只生成一张图片
                    if image_url:
                        print(f"图片URL: {image_url}")
                        return image_url
            return None
        return None

    except requests.exceptions.RequestException as e:
        print(f"查询图片生成任务 {task_id} 失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        return None
