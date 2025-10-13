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

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height)->dict:
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
    out = {} # {"lable":[abs_x1,abs_y1,abs_x2,abs_y2]}
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
      bbox = [abs_x1,abs_y1,abs_x2,abs_y2]
      # Draw the text
      assert "label" in bounding_box
      draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)
      out[bounding_box["label"]] = bbox

    # Display the image
    img.show()
    # Save image
    img.save("test.png")
    print("plt end")
    return out


def plot_points(im, text, input_width, input_height):
  """
  return point list [(x1,y1,"lable"),(x2,y2,"lable")...]
  """
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors

  # ---- 仅此处最小修改：支持多 <points> ----
  xml_text = text.replace('```xml', '').replace('```', '').strip()
  points_list = []  # [(x,y,label), ...]
  try:
    # 如果包含多个 <points>，用一个根包起来再解析
    if xml_text.count("<points") >= 1:
      import xml.etree.ElementTree as ET
      wrapped = f"<root>{xml_text}</root>"
      root = ET.fromstring(wrapped)
      for i, node in enumerate(root.iter("points")):
        x = node.attrib.get("x1")
        y = node.attrib.get("y1")
        if x is None or y is None: 
          continue
        label = node.attrib.get("alt") or (node.text.strip() if node.text else f"pt_{i+1}")
        points_list.append((float(x), float(y), label))
    else:
      # 兼容旧格式（单节点、成对属性）：沿用你原来的 decode_xml_points
      data = decode_xml_points(xml_text)
      if data is not None:
        # data['points'] 形如 [[x1,y1],[x2,y2],...]; data['phrase'] 为统一描述
        for i, p in enumerate(data['points']):
          if p and len(p) >= 2:
            points_list.append((float(p[0]), float(p[1]), data.get('phrase') or f"pt_{i+1}"))
  except Exception as e:
    # 解析失败时按原逻辑直接显示并返回
    print(e)
    img.show()
    return

  # 若无可绘制点，按原逻辑直接显示并返回
  if not points_list:
    img.show()
    return

  # 字体保持不变（不可用则抛异常按你原逻辑）
  font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

  # 绘制
  for i, (px, py, desc) in enumerate(points_list):
    color = colors[i % len(colors)]
    abs_x1 = float(px) / input_width * width
    abs_y1 = float(py) / input_height * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 + 8, abs_y1 + 6), str(desc), fill=color, font=font)

  img.save("point.png")
  img.show()
  return points_list

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



