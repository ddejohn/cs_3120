# ADAPTED FROM https://stackoverflow.com/a/55435714

import cv2
import numpy as np 
from keras.models import load_model


model = load_model("./final_model.h5")
drawing = False # true if mouse is pressed
posx, posy = None, None

# mouse callback function
def line_drawing(event, x, y, flags, param):
    global posx, posy, drawing

    def draw():
        cv2.line(
            img, (posx, posy), (x, y),
            color=(255,255,255),
            thickness=10
        )

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        posx, posy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            draw()
            posx, posy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        draw()


img = np.zeros((448,448,1))
cv2.namedWindow("draw a number")
cv2.setMouseCallback("draw a number", line_drawing)


while(1):
    cv2.imshow("draw a number", img)
    k = cv2.waitKey(1)
    if k & 0xFF == 27:
        break
    elif k & 0xFF == 13:
        X = cv2.resize(img/255.0, (28, 28), interpolation=cv2.INTER_AREA)
        X = X.reshape(1, 28, 28, 1)
        prediction = np.argmax(model.predict(X))
        print(f"\nLooks like a {prediction}!\n")
        img = np.zeros((448,448,1))


cv2.destroyAllWindows()
