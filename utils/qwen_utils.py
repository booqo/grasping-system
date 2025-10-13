# @title Plotting Util

# Get Noto JP font to display janapese characters
# !apt-get install fonts-noto-cjk  # For Noto Sans CJK JP

#!apt-get install fonts-source-han-sans-jp # For Source Han Sans (Japanese)

import json
import random
import io
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import json, re

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """
    print("begin")

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    img.show()
    # Save image
    img.save("test.png")
    print("plt end")


def plot_points(im, text, input_width, input_height):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors
  xml_text = text.replace('```xml', '')
  xml_text = xml_text.replace('```', '')
  data = decode_xml_points(xml_text)
  if data is None:
    img.show()
    return
  points = data['points']
  description = data['phrase']

  font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

  for i, point in enumerate(points):
    color = colors[i % len(colors)]
    abs_x1 = int(point[0])/input_width * width
    abs_y1 = int(point[1])/input_height * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)
  img.save("point.png")
  img.show()
  

# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

print("done")



def parse_agent_json(text: str):
    """
    处理模型返回的纯文本:
    1) 先直接 json.loads；
    2) 若失败，自动修复 function 数组中函数字符串里未转义的引号，例如：
       "vlm_move("请抓取…")" -> "vlm_move(\"请抓取…\")"
    3) 再次尝试 json.loads，最终返回 {"function": [...], "response": "..."}。
    """
    # 1) 尝试直接解析
    try:
        data = json.loads(text)
        return _normalize_agent_json(data)
    except Exception:
        pass

    # 2) 轻量修复：只在 "function": [ ... ] 这段内，把函数字符串里的内部引号转义
    def _fix_calls_inside_array(array_src: str) -> str:
        # 匹配形如 "fname(……)" 的“整体被双引号包裹”的函数字符串，
        # 再把括号内的双引号替换为 \"
        def _repl(m: re.Match) -> str:
            fname = m.group(1)
            args  = m.group(2).replace('"', r'\"')
            return f"\"{fname}({args})\""
        return re.sub(r'"([A-Za-z_][\w\.]*)\((.*?)\)"', _repl, array_src, flags=re.S)

    def _fix_text(t: str) -> str:
        # 只修复 function 数组，不碰其他字段
        pattern = r'("function"\s*:\s*\[)(.*?)(\])'
        def _wrap(m: re.Match) -> str:
            prefix, array_src, suffix = m.group(1), m.group(2), m.group(3)
            return prefix + _fix_calls_inside_array(array_src) + suffix
        return re.sub(pattern, _wrap, t, flags=re.S)

    repaired = _fix_text(text)

    # 3) 再次解析
    data = json.loads(repaired)
    return _normalize_agent_json(data)

def _normalize_agent_json(data: dict):
    """校验与规范化输出结构。"""
    if not isinstance(data, dict):
        raise ValueError("输出不是 JSON 对象。")

    fn_list = data.get("function", [])
    resp = data.get("response", "")

    if not isinstance(fn_list, list) or not all(isinstance(x, str) for x in fn_list):
        raise ValueError("'function' 必须为字符串数组。")
    if not isinstance(resp, str):
        raise ValueError("'response' 必须为字符串。")

    return {"function": fn_list, "response": resp}