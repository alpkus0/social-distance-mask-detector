import cv2
import math
from ultralytics import YOLO

model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
colors = {
    'with_mask': (0, 255, 0),  # Green
    'mask_weared_incorrect': (0, 255, 255),  # Yellow
    'without_mask': (0, 0, 255)  # Red
}

CONF_THRESHOLD = 0.5
SOCIAL_DISTANCE_THRESHOLD_CLOSE = 50  # Very close distance threshold (in pixels)
SOCIAL_DISTANCE_THRESHOLD = 150       # Regular social distance threshold (in pixels)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera is not working.")
        break

    results = model(frame)[0]

    people_centers = []
    people_boxes = []
    people_classes = []
    people_confs = []

    # Collect detected people
    for box in results.boxes:
        conf = box.conf[0]
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        cls_name = classes[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        people_centers.append((center_x, center_y))
        people_boxes.append((x1, y1, x2, y2))
        people_classes.append(cls_name)
        people_confs.append(conf)

    risk_indexes = set()

    # Social distance check
    if len(people_centers) > 1:
        for i in range(len(people_centers)):
            for j in range(i + 1, len(people_centers)):
                dist = math.sqrt(
                    (people_centers[i][0] - people_centers[j][0]) ** 2 +
                    (people_centers[i][1] - people_centers[j][1]) ** 2
                )

                mask_i = people_classes[i]
                mask_j = people_classes[j]

                # If both are wearing masks, warn only if very close
                if mask_i == 'with_mask' and mask_j == 'with_mask':
                    if dist < SOCIAL_DISTANCE_THRESHOLD_CLOSE:
                        risk_indexes.add(i)
                        risk_indexes.add(j)
                else:
                    # If at least one is unmasked or wearing incorrectly, use regular threshold
                    if dist < SOCIAL_DISTANCE_THRESHOLD:
                        risk_indexes.add(i)
                        risk_indexes.add(j)
    else:
        # No risk if only one person is detected
        risk_indexes = set()

    # Draw boxes and labels
    for idx, (x1, y1, x2, y2) in enumerate(people_boxes):
        cls_name = people_classes[idx]
        conf = people_confs[idx]

        if idx in risk_indexes:
            color = (0, 0, 255)  # Red for risky individuals
            label = f"Risk! {cls_name}"
        else:
            color = colors[cls_name]
            label = cls_name

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf * 100:.3f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Global social distance warning
    if len(risk_indexes) > 0:
        cv2.putText(frame, "SOCIAL DISTANCE VIOLATION!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Mask and Social Distance Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()
