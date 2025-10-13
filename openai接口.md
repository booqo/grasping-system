# openai接口类型说明
## 1. 可用字段

messages: 数组，每个元素为一条对话消息：

role: "system" | "user" | "assistant"

content: 两种写法其一

字符串（仅文本）；或

对象数组（多模态）：每个对象称为“内容块（content part）”

文本块：{"type": "text", "text": "<自然语言文本>" }

图片块：{"type": "image_url", "image_url": {"url": "<图片URL或本地路径或dataURL>"}}

## 2. 解析与约束
字符串 content 自动视为文本若某条消息的 content 是纯字符串，会被当作 {"type":"text","text":...} 处理。

图片来源（image_url.url）支持两种
本地路径：如 "/abs/path/img.jpg"、"./rel/path.png"（服务端会转换为 file://）
data: URL（base64）：如 data:image/jpeg;base64,<...>
不支持 直接的 http(s) 网络地址（默认会报错）。如需网络地址，请先在客户端下载成本地文件或转为 data: URL。
多内容块按顺序拼接同一条消息的 content 可交替包含文本与图片，顺序即语义。例如先给“任务要求文本”，紧跟“相关图片”，利于模型对齐。
角色用法
system：放“行为规范/全局约束”
user：放用户的提问、任务描述、图片等
assistant：放历史回复（可为空或仅文本）多轮对话时，按时间顺序排列 messages。
其他参数
max_tokens：生成上限，默认 512
temperature：采样温度（>0 启用采样），默认 0.2
stream：当前实现仅非流式（false）
## 3. 一些例子
### 3.1 文本only
```json
{
  "model": "Qwen2.5-VL-7B-Instruct",
  "messages": [
    {"role": "user", "content": "你好，做一个50字以内的自我介绍。"}
  ],
  "max_tokens": 128
}
```
### 3.2 单图+提示词
```json
{
  "model": "Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片。"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}}
      ]
    }
  ]
}
```
### 3.3 多图理解（同一轮消息内多张图）
```json
{
  "model": "Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "对比两张图片中的目标差异："},
        {"type": "image_url", "image_url": {"url": "/data/imgs/cam1.jpg"}},
        {"type": "image_url", "image_url": {"url": "/data/imgs/cam2.jpg"}}
      ]
    }
  ]
}
```
### 3.4 文字-图片-文字 交错（逐步指示）
```json
{
  "model": "Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "第一步：识别图片里的机械臂型号。"},
        {"type": "image_url", "image_url": {"url": "/data/robot_arm.jpg"}},
        {"type": "text", "text": "第二步：给出臂长和自由度。"}
      ]
    }
  ]
}
```
### 3.5 多轮对话（含历史 assistant 回复）
```json
{
  "model": "Qwen2.5-VL-7B-Instruct",
  "messages": [
    {"role": "system", "content": "你是专业的多模态视觉助手，回答简洁、准确。"},
    {"role": "user", "content": "这张图中的零件是什么？"},
    {"role": "assistant", "content": "看起来是联轴器，但不确定规格。"},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "给你更清晰的一张图，请确认型号。"},
        {"type": "image_url", "image_url": {"url": "/data/coupler_closeup.png"}}
      ]
    }
  ],
  "max_tokens": 256,
  "temperature": 0.2
}
```
### 3.6 本地路径与 dataURL 的对照
本地文件（服务端可访问）：
```json
{"type":"image_url","image_url":{"url":"/abs/path/to/part.jpg"}}
```
data URL（跨机/远程最稳妥）：
```json
{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAA..."}}
```
## 设计要点与建议

跨机调用时优先用 data: URL：若客户端与 API 服务不在同一台机器，本地路径在服务端不可见，建议把图片转为 base64 的 data: URL 再发送。例如：
```python
with open("detection_visualization.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
data_url = f"data:image/jpeg;base64,{b64}"
```

内容块顺序即语义：建议“文本指令→图片→补充文本”这种结构，能显著提升指令对齐与可解释性。

detail 等扩展字段：OpenAI 的 image_url 可带 detail（high/low），当前实现会忽略该字段，不影响使用。

禁止直接 http(s) 图片：如确需网络图片，请在客户端先下载或自行扩展服务端逻辑（下载到临时文件）。

大小/时延：data: URL 体积较大时，请适当提高超时或缩放图片；max_tokens 不要设得过大。

系统提示（system）：用于设定风格、边界与上下文假设，尤其在复杂任务（如抓取位姿解析）中建议使用。