import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for hanndLM in result.multi_hand_landmarks:
            for id , lm in enumerate(hanndLM.landmark):
                h ,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)

            if id  == 4:
                cv2.circle(img,(cx,cy), 20, (225,223,0),cv2.FILLED)

        mpDraw.draw_landmarks(img,hanndLM,mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,
                (225,10,20),3)



    cv2.imshow("Image", img)
    cv2.waitKey(1)



