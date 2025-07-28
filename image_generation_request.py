import requests
import json

def generate_image(prompt: str):
    print("开始图片生成请求...")

    # 从api_token.txt读取鉴权信息
    with open('api_token.txt', 'r') as f:
        api_token = f.read().strip()

    # API请求的URL
    url = "https://api-beijing.klingai.com/v1/images/generations" # 实际的API地址

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    # 请求体
    payload = {
        "model_name": "kling-v1",  # 可选，模型名称
        "prompt": prompt,  # 必须，正向文本提示词
        "negative_prompt": "",  # 可选，负向文本提示词
        "image": "",  # 可选，参考图像（Base64编码或URL）
        "image_reference": "",  # 可选，图片参考类型（subject, face）
        "image_fidelity": 0.5,  # 可选，生成过程中对用户上传图片的参考强度
        "human_fidelity": 0.45,  # 可选，面部参考强度
        "n": 1,  # 可选，生成图片数量
        "aspect_ratio": "16:9",  # 可选，生成图片的画面纵横比
        "callback_url": ""  # 可选，本次任务结果回调通知地址
    }

    # 发送POST请求
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)
        response.raise_for_status() # 检查HTTP请求是否成功

        print("图片生成请求成功！")
        response_data = response.json()
        print("响应内容:", json.dumps(response_data, indent=4, ensure_ascii=False))

        # 提取图片URL
        if response_data and response_data.get("code") == 0 and response_data.get("data"):
            task_id = response_data["data"].get("task_id")
            if task_id:
                print(f"图片生成任务ID: {task_id}")
                # 这里需要一个机制来查询任务状态并获取图片URL
                # 暂时返回task_id，后续通过查询接口获取图片URL
                return task_id
        return None

    except requests.exceptions.RequestException as e:
        print(f"图片生成请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        return None
