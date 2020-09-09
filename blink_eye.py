import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
first_read = True

# starting the video capture
cap = cv2.VideoCapture(0)
ret, img = cap.read()
cv2.resizeWindow('img', 1000, 1000)
while ret:

    ret, img = cap.read()
    # converting the recorded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # applying the filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)
    face = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))

    if len(face) > 0:
        for (x, y, w, h) in face:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # roi_face is face which is input to eye classifier
            roi_face = gray[y:y + h, x:x + w]
            roi_face_clr = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

            # Examining the length of eye object for eyes
            if len(eyes) >= 1:
                # check if program is running for detection
                if first_read:
                    cv2.putText(img,
                                "Eye detected press S to begin",
                                (70, 70),
                                cv2.FONT_ITALIC, 1,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(img,
                                "Eyes open!",
                                (70, 70),
                                cv2.FONT_ITALIC, 1,
                                (255, 255, 255), 2)
            else:
                if first_read:
                    cv2.putText(img,
                                "No Eyes Detected", (70, 70),
                                cv2.FONT_ITALIC, 1,
                                (0, 0, 255), 2)
                else:
                    print("Blink detected!")
                    cv2.waitKey(3000)
                    first_read = True
    else:
        cv2.putText(img,
                    "No face detected",
                    (100, 100),
                    cv2.FONT_ITALIC, 1,
                    (0, 255, 0), 2)

    # Controlling the algorithm with keys
    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if a == ord('q'):
        break
    elif a == ord('s') and first_read:
        first_read = False
cap.release()
cv2.destroyAllWindows()
