import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont


def det_visualizer(preds, img, quads, vis_heatmap=False, line_color=(0, 255, 0)):
    preds = preds["thresh_binary"][0]
    binary = preds.detach().cpu().numpy()
    out = img.copy()
    h, w = out.shape[:2]
    binary = binary.squeeze(0)
    binary = (binary * 255).astype(np.uint8)
    if vis_heatmap:
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(binary, cv2.COLORMAP_JET)
        out = cv2.addWeighted(out, 0.5, heatmap, 0.5, 0)

    for quad in quads:
        quad = np.array(quad).astype(np.int32)
        out = cv2.polylines(out, [quad], True, line_color, 1)
    return out


def rec_visualizer(
    img,
    preds,
    quads,
    font_size=12,
    font_color=(255, 0, 0),
    font_path="resource/MPLUS1p-Medium.ttf",
):
    out = img.copy()
    pillow_img = Image.fromarray(out)
    draw = ImageDraw.Draw(pillow_img)

    for pred, quad in zip(preds, quads):
        quad = np.array(quad).astype(np.int32)
        word_width = np.linalg.norm(quad[0] - quad[1])
        word_height = np.linalg.norm(quad[1] - quad[2])
        font = ImageFont.truetype(font_path, font_size)

        direction = "horizontal"
        if word_width * 2 < word_height:
            direction = "vertical"

        if direction == "horizontal":
            x_offset = 0
            y_offset = -font_size

            pos_x = quad[0][0] + x_offset
            pox_y = quad[0][1] + y_offset
            draw.text((pos_x, pox_y), pred, font=font, fill=font_color)
        else:
            x_offset = -font_size
            y_offset = 0

            pos_x = quad[0][0] + x_offset
            pox_y = quad[0][1] + y_offset
            draw.text(
                (pos_x, pox_y),
                pred,
                font=font,
                fill=font_color,
                direction="ttb",
            )

    out = np.array(pillow_img)
    return out
