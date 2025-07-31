# AI Flow Keling 项目

本项目包含一系列用于AI工作流的Python脚本，主要涉及图像生成、图像到视频转换、以及相关的API交互。

## 文件说明

- `demo_classification.py`: 演示分类功能的脚本。
- `generate_token.py`: 用于生成API访问令牌。
- `image_generation_request.py`: 发送图像生成请求的脚本。
- `image_renamer.py`: 图像文件重命名工具。
- `image2video_request.py`: 发送图像到视频转换请求的脚本。
- `main_workflow.py`: 主工作流程脚本，可能整合了其他功能。
- `query_image_generations.py`: 查询图像生成任务状态的脚本。
- `query_image2video_task.py`: 查询图像到视频转换任务状态的脚本。
- `test_gemini_api.py`: 测试Gemini API的脚本。
- `config.json`: 项目配置文件，包含API密钥等敏感信息（已在.gitignore中排除）。
- `流程.txt`: 项目流程或说明文档。

## 快速开始

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/windy-freedom/ai-flow-keling.git
    cd ai-flow-keling
    ```
2.  **安装依赖**:
    ```bash
    # 根据项目实际依赖安装，例如：
    # pip install requests
    # pip install google-generativeai
    ```
3.  **配置API密钥**:
    创建一个 `api_token.txt` 文件，并将您的API密钥放入其中。
    （此文件已在 `.gitignore` 中排除，不会被提交到版本控制。）
4.  **运行脚本**:
    根据需要运行相应的Python脚本。

## 注意事项

- `api_token.txt` 和 `downloads/` 目录已在 `.gitignore` 中排除，请勿将敏感信息和生成的图片提交到版本控制。
