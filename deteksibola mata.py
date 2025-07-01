import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi tidak berubah
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Fungsi tidak berubah
def get_gaze_direction(eye_landmarks):
    left_point_x = eye_landmarks[0][0]
    right_point_x = eye_landmarks[1][0]
    center_iris_x = eye_landmarks[2][0]
    
    eye_width = right_point_x - left_point_x
    if eye_width == 0:
        return "TENGAH", 0.5

    gaze_ratio = (center_iris_x - left_point_x) / eye_width

    if gaze_ratio < 0.40:
        direction = "KIRI"
    elif gaze_ratio > 0.60:
        direction = "KANAN"
    else:
        direction = "TENGAH"
        
    return direction, gaze_ratio


# Mulai menangkap video
cap = cv2.VideoCapture(2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # =================== PERUBAHAN DI SINI ===================
            # Kita sekarang menggunakan MATA KANAN (yang ada di kiri layar)
            # 33: Sudut kiri, 133: Sudut kanan, 473: Tengah Iris
            eye_lm = [face_landmarks.landmark[33], face_landmarks.landmark[133], face_landmarks.landmark[473]]
            # =========================================================
            
            eye_coords = [(int(p.x * img_w), int(p.y * img_h)) for p in eye_lm]

            direction, gaze_ratio = get_gaze_direction(eye_coords)

            # Visualisasi Debugging (tidak berubah)
            cv2.circle(image, eye_coords[0], 3, (0, 255, 255), -1) 
            cv2.circle(image, eye_coords[1], 3, (255, 0, 255), -1) 
            cv2.circle(image, eye_coords[2], 3, (255, 255, 0), -1) 
            
            eye_width = eye_coords[1][0] - eye_coords[0][0]
            if eye_width != 0:
                line_40_percent = eye_coords[0][0] + int(0.40 * eye_width)
                line_60_percent = eye_coords[0][0] + int(0.60 * eye_width)
                eye_y_level = eye_coords[0][1]
                
                cv2.line(image, (line_40_percent, eye_y_level - 5), (line_40_percent, eye_y_level + 5), (0, 255, 0), 1)
                cv2.line(image, (line_60_percent, eye_y_level - 5), (line_60_percent, eye_y_level + 5), (0, 255, 0), 1)
            
            cv2.putText(image, f"Arah: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Rasio: {gaze_ratio:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Deteksi Arah Pandang Mata', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()