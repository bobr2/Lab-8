import cv2
import numpy as np
import time


def image_processing():
    img = cv2.imread('images/variant-7.jpg')
    h, w = img.shape[:2]

    hor_im = cv2.flip(img, 1)
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
    rotated = cv2.warpAffine(hor_im, M, (w, h))
    
    cv2.imshow('rotated', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_processing():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Поиск кругов
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=60, param2=40, minRadius=20, maxRadius=50
        )

        # Отрисовка центра кадра
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Используем только первый круг
            cx, cy, radius = circles[0][0]

            # Расстояние в пикселях до центра кадра
            pixel_distance = np.linalg.norm([cx - center_x, cy - center_y])

            # Отрисовка круга и центра
            cv2.circle(frame, (cx, cy), radius, (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Линия и текст расстояния
            cv2.line(frame, (center_x, center_y), (cx, cy), (0, 255, 255), 1)
            cv2.putText(frame, f"{int(pixel_distance)} px",
                        (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

        cv2.imshow("Single Circle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #image_processing()
    video_processing()
