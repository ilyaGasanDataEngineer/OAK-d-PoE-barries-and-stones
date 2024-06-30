import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def putText(img, text, position, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font_path = "/home/gasan/PycharmProjects/depthAI_main/test/gen2-yolo/device-decoding/fonts/font_kirill.ttf"  # Укажите путь к вашему шрифту
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    img = np.array(img_pil)
    return img

cap = cv2.VideoCapture('result_vid3.mp4')
if (cap.isOpened() == False):
    print('error')
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cv2.waitKey(2555904)
