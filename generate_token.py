import time
import jwt

ak = "ACPp3YnM9eyJHHH99FBY9pfMhBdM9BYM" # fill access key
sk = "Ta9KADtGyBatYtFhrTmmHRLtBeDdbNyf" # fill secret key

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
with open('api_token.txt', 'w') as f:
    f.write(authorization)
print("API Token 已生成并保存到 api_token.txt")
