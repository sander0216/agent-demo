from openai import OpenAI

# 1. 初始化客户端
# 这里我们虽然用的是OpenAI这个库，但是连的是DeepSeek的服务器
client = OpenAI(
    api_key="sk-a821aba999ce40cc9c8349cf7ac2871d",  # 比如 "sk-abc123456..."
    base_url="https://api.deepseek.com"  # 这是一个通用的地址
)

print("正在连接模型，请稍等...")

# 2. 发送请求
# 这里是核心：你把消息发给模型，模型给你回信
response = client.chat.completions.create(
    model="deepseek-chat",  # 指定用哪个模型
    messages=[
        # system: 给模型的人设（背景设定）
        {"role": "system", "content": "你是一个有用的AI助手"},
        # user: 用户（也就是你）说的话
        {"role": "user", "content": "你好，这是我第一次写代码调用你！"},
    ],
    stream=False
)

# 3. 打印结果
# 模型的回复藏在 response.choices[0].message.content 里面
print("="*20)
print("模型回复：")
print(response.choices[0].message.content)
print("="*20)