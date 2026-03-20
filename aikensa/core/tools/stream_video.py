import cv2
import sys

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)  # try 0 first

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)

print("Camera opened successfully. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame.")
        break

    cv2.imshow("Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
