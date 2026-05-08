import cv2

face_cascades = cv2.CascadeClassifier(r"C:\Users\SUHANI\.vscode\openCV\project\haarcascade_frontalface_default.xml")
eye_cascades = cv2.CascadeClassifier(r"C:\Users\SUHANI\.vscode\openCV\project\haarcascade_eye.xml")
smile_cascades = cv2.CascadeClassifier(r"C:\Users\SUHANI\.vscode\openCV\project\haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Couldn't read frame.")
        break
    # Flip camera for mirror effect
    frame = cv2.flip(frame, 1)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
# detect face
    faces = face_cascades.detectMultiScale(gray, 1.1, 5)


    for (x, y, w, h) in faces:
        # x=left distance,y=botton distance,w=width of the image,h=height of the image
        cv2.rectangle(blurred, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(blurred,"FACE DETECTED",(x, y-10),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,255,0),2)
        # (x,y)=left most distance of image
        # (x+w, y+h)=botton-right distance of the image

        roi_gray = gray[y:y+h, x:x+w]
        # slicling for easy detection so we not need to search whole face.work on gray image
        roi_color = blurred[y:y+h, x:x+w]
        # slicling to draw rectangle on colored image 
        extra_h = int(h * 0.5)

        y2 = min(y + h + extra_h, frame.shape[0])

        blurred[y:y2, x:x+w] = frame[y:y2, x:x+w]

        # Detect smile
        smiles = smile_cascades.detectMultiScale(roi_gray, 1.7, 20)
        for (sx, sy, sw, sh) in smiles:
           if len(smiles)>0: 
             cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255,0, 255), 2)

        # Detect eyes
        eyes = eye_cascades.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
          cv2.rectangle(roi_gray, (ex, ey), (ex+ew, ey+eh), (255,0, 255), 2)
          if len(smiles) > 0:
            mood = "MOOD:Happy"
            mood_color = (0, 255, 0)

          elif len(eyes) <=1:
            mood = "MOOD:Sleepy"
            mood_color = (255, 0, 0)

          else:
            mood = "MOOD:Neutral"
            mood_color = (0, 255, 255)

        # Mood Text
        cv2.putText(blurred,mood,(x, y+h+30),cv2.FONT_HERSHEY_SIMPLEX,0.8,mood_color,2)

    cv2.imshow("Smart Mood Detection", blurred)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("QUITTING")
        break

cap.release()
cv2.destroyAllWindows()