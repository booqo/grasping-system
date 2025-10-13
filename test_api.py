import base64
from openai import OpenAI



client = OpenAI(base_url="http://10.18.138.8:8000/v1", api_key="EMPTY")


resp = client.chat.completions.create(
    model="Qwen2.5-VL-7B-Instruct",
    messages=[{"role": "user", "content": "你好，简单自我介绍一下。"}],
    temperature=0.2,
    max_tokens=256
)
print(resp.choices[0].message.content)



client = OpenAI(base_url="http://10.18.138.8:8000/v1", api_key="EMPTY")

with open("detection_visualization.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
data_url = f"data:image/jpeg;base64,{b64}"

messages = [
    {"role": "user",
     "content": [
         {"type": "text", "text": "请描述这张图片。"},
         {"type": "image_url", "image_url": {"url": data_url}}
     ]}
]

resp = client.chat.completions.create(model="Qwen2.5-VL-7B-Instruct",
                                      messages=messages,
                                      max_tokens=128)
print(resp.choices[0].message.content)