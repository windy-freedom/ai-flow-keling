import requests
import json
import base64
import os

def generate_video(image_url: str, prompt: str = ""):
    print("开始视频生成请求...")

    # 从api_token.txt读取鉴权信息
    with open('api_token.txt', 'r') as f:
        api_token = f.read().strip()

    # API请求的URL
    url = 'https://api-beijing.klingai.com/v1/videos/image2video' # 假设image2video的API也在api-beijing.klingai.com下
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    data = {
        "model_name": "kling-v1",
        "mode": "pro",
        "duration": "5",
        "image": image_url, # 使用传入的图片URL
        "prompt": prompt, # 继承图片生成的提示词
        "cfg_scale": 0.5,
        "static_mask": "https://h2.inkwai.com/bs2/upload-ylab-stunt/ai_portal/1732888177/cOLNrShrSO/static_mask.png",
        "dynamic_masks": [
          {
            "mask": "https://h2.inkwai.com/bs2/upload-ylab-stunt/ai_portal/1732888130/WU8spl23dA/dynamic_mask_1.png",
            "trajectories": [
              {"x":279,"y":219},{"x":417,"y":65}
            ]
          }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
        response.raise_for_status()  # 如果请求失败，抛出HTTPError异常
        print("视频生成请求成功！")
        response_data = response.json()
        print("响应内容:", json.dumps(response_data, indent=4, ensure_ascii=False))

        if response_data and response_data.get("code") == 0 and response_data.get("data"):
            task_id = response_data["data"].get("task_id")
            if task_id:
                print(f"视频生成任务ID: {task_id}")
                return task_id
        return None

    except requests.exceptions.RequestException as e:
        print(f"视频生成请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None
