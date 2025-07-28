import time
import jwt
import os

# 从环境变量中读取 AK 和 SK
ak = os.environ.get("AK")
sk = os.environ.get("SK")

if not ak or not sk:
    raise ValueError("环境变量 AK 和 SK 必须设置")

def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 43200, # The valid time, in this example, represents the current time+43200s(12h)
        "nbf": int(time.time()) - 5 # The time when it starts to take effect, in this example, represents the current time minus 5s
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token

authorization = encode_jwt_token(ak, sk)
print("API Token 已生成")
# 如果需要将 token 保存到文件，请在调用此脚本的地方处理
# with open('api_token.txt', 'w') as f:
#     f.write(authorization)
