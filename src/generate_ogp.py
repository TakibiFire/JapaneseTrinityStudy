"""
OGP画像生成スクリプト
docs/*.md の各ファイルに対して、タイトルスライドのようなOGP画像を生成し、
docs/imgs/${doc}/thumbnail.jpg に保存します。
"""

import argparse
import glob
import os
import re

from PIL import Image, ImageDraw, ImageFont

# 設定
IMAGE_SIZE = (1200, 630)
BACKGROUND_COLOR = (255, 255, 255)
BORDER_COLOR = (0, 148, 133)
BORDER_THICKNESS_PERCENT = 0.05

TITLE_FONT_SIZE = 60
SUB_FONT_SIZE = 40
FONT_PATH = os.path.expanduser("~/Library/Fonts/NotoSansJP-Bold.otf")

SITE_TITLE_1 = "日本版FIRE後の取り崩し戦略"
SITE_TITLE_2 = "〜 4%ルールを信じるな 〜"
AUTHOR_NAME = "たきび FIRE"
LOGO_PATH = "docs/imgs/takibi.png"


def get_title_from_md(file_path):
  """Markdownファイルから最初のH1タイトルを抽出する"""
  with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
      match = re.match(r"^#\s+(.+)$", line)
      if match:
        title = match.group(1).strip()
        # "〜" を含む場合の処理。
        # "A 〜 B" を "A<wbr/>〜 B 〜" に変換する。
        # すでに <wbr/> がある場合も考慮して、一旦 "〜" の前のスペースやタグを整理する。
        if "〜" in title:
          # "〜" の前の <wbr/> やスペースを消してから、"<wbr/>〜 " を挿入する
          title = re.sub(r"(?:<wbr/>|\s)*〜\s*", "<wbr/>〜 ", title)
          # 末尾に " 〜" がなければ追加する
          if not title.endswith(" 〜"):
            title += " 〜"
        return title
  return os.path.basename(file_path).replace(".md", "")


def wrap_text(text, font, max_width):
  """指定された幅に合わせてテキストを改行する。 <wbr/> タグがあればそこで優先的に改行する。"""
  segments = text.split("<wbr/>")
  all_lines = []

  for segment in segments:
    if not segment:
      continue
    # セグメントが幅に収まるかチェック
    bbox = font.getbbox(segment)
    if bbox[2] - bbox[0] <= max_width:
      all_lines.append(segment)
    else:
      # 収まらない場合は文字単位で分割
      lines = []
      current_line = ""
      for char in segment:
        test_line = current_line + char
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        if width <= max_width:
          current_line = test_line
        else:
          if current_line:
            lines.append(current_line)
          current_line = char
      if current_line:
        lines.append(current_line)
      all_lines.extend(lines)
  return all_lines


def generate_ogp(title, output_path, is_index=False):
  """OGP画像を生成する"""
  # 表示用のタイトルからはタグを除く
  display_title = title.replace("<wbr/>", "")
  img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
  draw = ImageDraw.Draw(img)

  # 枠の描画
  w, h = IMAGE_SIZE
  bt = int(h * BORDER_THICKNESS_PERCENT)

  # Border: Top, Bottom, Right, Left
  draw.rectangle([0, 0, w, bt], fill=BORDER_COLOR)  # Top
  draw.rectangle([0, h - bt, w, h], fill=BORDER_COLOR)  # Bottom
  draw.rectangle([0, 0, bt, h], fill=BORDER_COLOR)  # Left
  draw.rectangle([w - bt, 0, w, h], fill=BORDER_COLOR)  # Right

  # フォントの読み込み
  title_font_size = 70 if is_index else TITLE_FONT_SIZE
  try:
    title_font = ImageFont.truetype(FONT_PATH, title_font_size)
    sub_font = ImageFont.truetype(FONT_PATH, SUB_FONT_SIZE)
  except Exception as e:
    print(f"Font loading failed: {e}")
    # フォールバック (macOSの他のフォントなど)
    title_font = ImageFont.load_default()
    sub_font = ImageFont.load_default()

  # メインタイトルの描画 (50%, 30% if not index else 50%, 40%)
  title_y_center = 0.4 if is_index else 0.3
  max_title_width = w * 0.8
  wrapped_title_lines = wrap_text(title, title_font, max_title_width)
  print(f"Title: {display_title}")
  print(f"  Wrapped: {wrapped_title_lines}")

  total_title_height = 0
  line_heights = []
  line_gaps = []
  for line in wrapped_title_lines:
    bbox = title_font.getbbox(line)
    line_height = bbox[3] - bbox[1]
    line_heights.append(line_height)
    # 行間を 1.5x に設定 (行高の 50% を追加)
    line_gap = int(line_height * 0.5)
    line_gaps.append(line_gap)
    total_title_height += line_height + line_gap

  current_y = h * title_y_center - total_title_height / 2
  for i, line in enumerate(wrapped_title_lines):
    bbox = title_font.getbbox(line)
    line_width = bbox[2] - bbox[0]
    draw.text(((w - line_width) / 2, current_y),
              line,
              font=title_font,
              fill=(0, 0, 0))
    current_y += line_heights[i] + line_gaps[i]

  if not is_index:
    # サイトタイトルの描画 (2行)
    # Line 1 (50%, 60%)
    bbox = sub_font.getbbox(SITE_TITLE_1)
    sw, sh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - sw) / 2, h * 0.6 - sh / 2),
              SITE_TITLE_1,
              font=sub_font,
              fill=(50, 50, 50))

    # Line 2 (50%, 70%)
    bbox = sub_font.getbbox(SITE_TITLE_2)
    sw, sh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - sw) / 2, h * 0.7 - sh / 2),
              SITE_TITLE_2,
              font=sub_font,
              fill=(50, 50, 50))

    # 著者名の描画 (50%, 80%)
    author_y = 0.8
  else:
    # index.md の場合はサイトタイトルを隠し、著者名を上げる
    author_y = 0.75

  bbox = sub_font.getbbox(AUTHOR_NAME)
  aw, ah = bbox[2] - bbox[0], bbox[3] - bbox[1]
  draw.text(((w - aw) / 2, h * author_y - ah / 2),
            AUTHOR_NAME,
            font=sub_font,
            fill=(50, 50, 50))

  # ロゴの描画
  if os.path.exists(LOGO_PATH):
    logo = Image.open(LOGO_PATH).convert("RGBA")
    # ロゴのリサイズ (適当なサイズに)
    logo.thumbnail((150, 150))
    lw, lh = logo.size
    # (85%, 80%) に配置
    img.paste(logo, (int(w * 0.85 - lw / 2), int(h * 0.8 - lh / 2)), logo)

  # 保存
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  # JPGとして保存するためにRGBに変換 (既にRGBだが念のため)
  img.save(output_path, "JPEG", quality=90)


def main():
  parser = argparse.ArgumentParser(
      description="Generate OGP images for Markdown files.")
  parser.add_argument(
      "--files",
      type=str,
      help="Comma-separated list of Markdown files to process.")
  args = parser.parse_args()

  if args.files:
    md_files = [f.strip() for f in args.files.split(",")]
  else:
    md_files = glob.glob("docs/*.md")

  for md_file in md_files:
    if not os.path.exists(md_file):
      print(f"File not found: {md_file}")
      continue

    basename = os.path.basename(md_file)
    doc_name = basename.replace(".md", "")
    output_path = f"docs/imgs/{doc_name}/thumbnail.jpg"
    is_index = (basename == "index.md")

    title = get_title_from_md(md_file)
    generate_ogp(title, output_path, is_index=is_index)
    print(f"Generated: {output_path}\n")


if __name__ == "__main__":
  main()
