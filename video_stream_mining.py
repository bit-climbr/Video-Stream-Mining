import cv2
import numpy as np
import pandas as pd
from datetime import datetime

video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

event_log = []

frame_count = 0
motion_threshold = 800 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    

    frame = cv2.resize(frame, (640, 480))


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    kernel = np.ones((5,5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    motion_pixels = np.sum(fgmask == 255)

    if motion_pixels > motion_threshold:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_log.append({
            "Frame": frame_count,
            "Timestamp": timestamp,
            "Motion_Pixels": motion_pixels,
            "Event": "High Motion Detected"
        })

        cv2.putText(frame, "Anomaly Detected!", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255), 3)

    cv2.imshow("Video Stream", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(event_log)
df.to_csv("event_metadata.csv", index=False)

print("Processing Completed.")
print("Metadata saved to event_metadata.csv")