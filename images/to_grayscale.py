import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def putText(img, text, position, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font_path = "/media/ilyagasan/disk_B/projects/diplom/fonts/font_kirill.ttf"  # Укажите путь к вашему шрифту
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    img = np.array(img_pil)
    return img

def nothing(x):
    pass

# Создание окна с ползунками
cv2.namedWindow('Trackbars')

# Создание ползунков для настройки цветовых границ
cv2.createTrackbar('Lower H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('Lower S', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('Lower V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'Trackbars', 80, 179, nothing)
cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Trackbars', 80, 255, nothing)

# Загрузка изображения
frame1 = cv2.imread('/media/ilyagasan/disk_B/projects/diplom/img.png')
frame1 = cv2.GaussianBlur(frame1, (5, 5), 0)

while True:
    # Получение значений с ползунков
    l_h = cv2.getTrackbarPos('Lower H', 'Trackbars')
    l_s = cv2.getTrackbarPos('Lower S', 'Trackbars')
    l_v = cv2.getTrackbarPos('Lower V', 'Trackbars')
    u_h = cv2.getTrackbarPos('Upper H', 'Trackbars')
    u_s = cv2.getTrackbarPos('Upper S', 'Trackbars')
    u_v = cv2.getTrackbarPos('Upper V', 'Trackbars')

    # Удаление зеленого цвета
    l_green = np.array([l_h, l_s, l_v])
    u_green = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(frame1, l_green, u_green)
    mask_inv = cv2.bitwise_not(mask)

    frame1_no_green = cv2.bitwise_and(frame1, frame1, mask=mask_inv)

    # Преобразование изображения в градации серого
    gray1 = cv2.cvtColor(frame1_no_green, cv2.COLOR_BGR2GRAY)

    # Пороговая обработка
    ret, thresh = cv2.threshold(gray1, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Создание пустого изображения для рисования контуров
    contour_image = np.zeros_like(frame1_no_green)

    # Рисование контуров и вычисление площади
    for contour in contours:
        # Рисование контура
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 3)

        # Вычисление площади контура
        area = cv2.contourArea(contour)

        # Вычисление центра масс контура
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = 0, 0

        # Вывод площади в центре контура
        contour_image2 = putText(contour_image, f'5.22 см²', (cX - 20, cY), 20, (255, 255, 255))

    # Отображение изображений
    cv2.imshow('frame1_no_green', frame1_no_green)
    cv2.imshow('contours', contour_image)
    cv2.imshow('contours2', contour_image2)
    cv2.imshow('gray', gray1)
    # Ожидание нажатия клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
