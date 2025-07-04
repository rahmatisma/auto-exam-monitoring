import cv2
import mediapipe as mp
import numpy as np

#nisialisasi face mesh dari MediaPipe untuk mendeteksi titik-titik wajah
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


calibrated_center = None


# Fungsi tidak berubah
def get_gaze_direction(left_eye_landmarks, right_eye_landmarks):
    # Gaze kiri
    left_width = left_eye_landmarks[1][0] - left_eye_landmarks[0][0]
    left_ratio = (left_eye_landmarks[2][0] - left_eye_landmarks[0][0]) / left_width if left_width != 0 else 0.5

    # Gaze kanan
    right_width = right_eye_landmarks[1][0] - right_eye_landmarks[0][0]
    right_ratio = (right_eye_landmarks[2][0] - right_eye_landmarks[0][0]) / right_width if right_width != 0 else 0.5

    # Ambil rata-rata dari kedua mata
    gaze_ratio = (left_ratio + right_ratio) / 2.0

    if gaze_ratio < 0.45:
        direction = "KIRI"
    elif gaze_ratio > 0.55:
        direction = "KANAN"
    else:
        direction = "TENGAH"
        
    return direction, gaze_ratio



def detect_head_turn(face_landmarks, img_w, img_h):
    nose_tip = face_landmarks.landmark[1]
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]

    nose_x = int(nose_tip.x * img_w)
    left_x = int(left_cheek.x * img_w)
    right_x = int(right_cheek.x * img_w)

    dist_left = abs(nose_x - left_x)
    dist_right = abs(nose_x - right_x)

    ratio = dist_left / dist_right if dist_right != 0 else 1

        #  mirroring arah kepala 
    if ratio > 1.2:
        direction = "KE KANAN" 
    elif ratio < 0.8:
            direction = "KE KIRI"  
    else:
        direction = "TENGAH"
       

    angle_diff = abs(dist_left - dist_right)
    return direction, angle_diff

# Mulai menangkap video
cap = cv2.VideoCapture(0)

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
            try:
                # Landmark mata kiri: sudut kiri (33), sudut kanan (133), iris (468)
                left_eye_lm = [face_landmarks.landmark[33], face_landmarks.landmark[133], face_landmarks.landmark[468]]

                # Landmark mata kanan: sudut kanan (362), sudut kiri (263), iris (473)
                right_eye_lm = [face_landmarks.landmark[362], face_landmarks.landmark[263], face_landmarks.landmark[473]]

                # Konversi ke koordinat piksel
                left_eye_coords = [(int(p.x * img_w), int(p.y * img_h)) for p in left_eye_lm]
                right_eye_coords = [(int(p.x * img_w), int(p.y * img_h)) for p in right_eye_lm]

                # Hitung arah pandangan
                direction, gaze_ratio = get_gaze_direction(left_eye_coords, right_eye_coords)

                # Visualisasi titik-titik mata kiri dan kanan
                for pt in left_eye_coords:
                    cv2.circle(image, pt, 3, (0, 255, 255), -1)
                for pt in right_eye_coords:
                    cv2.circle(image, pt, 3, (255, 0, 255), -1)

                # Garis batas 40% dan 60% pada mata kiri
                left_width = left_eye_coords[1][0] - left_eye_coords[0][0]
                if left_width != 0:
                    l40 = left_eye_coords[0][0] + int(0.40 * left_width)
                    l60 = left_eye_coords[0][0] + int(0.60 * left_width)
                    y_level = left_eye_coords[0][1]
                    cv2.line(image, (l40, y_level - 5), (l40, y_level + 5), (0, 255, 0), 1)
                    cv2.line(image, (l60, y_level - 5), (l60, y_level + 5), (0, 255, 0), 1)

                # Garis batas pada mata kanan
                right_width = right_eye_coords[1][0] - right_eye_coords[0][0]
                if right_width != 0:
                    r40 = right_eye_coords[0][0] + int(0.40 * right_width)
                    r60 = right_eye_coords[0][0] + int(0.60 * right_width)
                    y_level = right_eye_coords[0][1]
                    cv2.line(image, (r40, y_level - 5), (r40, y_level + 5), (0, 255, 0), 1)
                    cv2.line(image, (r60, y_level - 5), (r60, y_level + 5), (0, 255, 0), 1)

                # Tampilkan arah dan rasio
                cv2.putText(image, f"Arah: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f"Rasio: {gaze_ratio:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except IndexError:
                continue
            
              #  Deteksi arah kepala 
            head_direction, angle_diff = detect_head_turn(face_landmarks, img_w, img_h)
            cv2.putText(image, f"Kepala: {head_direction}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            #  Estimasi arah pandangan dengan offset pipi
            nose_tip = face_landmarks.landmark[1]
            center_x = int(nose_tip.x * img_w)
            center_y = int(nose_tip.y * img_h)
            radius = 50

            # Gunakan pipi kiri dan kanan untuk hitung offset
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]
            head_center_x = int((left_cheek.x + right_cheek.x) / 2 * img_w)
            shift_x = (center_x - head_center_x) * 1  # perbesar pengaruh

            # Titik arah hidung (bergerak sesuai arah pandangan)
            titik_merah = (int(center_x + shift_x), int(center_y))

            # Gambar lingkaran yang mengikuti hidung
            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)   # LINGKARAN HIJAU bergerak
            cv2.circle(image, titik_merah, 5, (0, 0, 255), -1)               #  TITIK MERAH arah pandang

            # Deteksi kecurangan jika titik arah pandang keluar dari zona aman
            dx = titik_merah[0] - center_x
            dy = titik_merah[1] - center_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance > radius:
                if abs(dx) > abs(dy):
                    status = "Menyontek (Kiri/Kanan)"
                else:
                    status = "Menyontek (Atas/Bawah)"
                cv2.putText(image, status, (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
      
            
             # Kalibrasi posisi awal hidung
            if calibrated_center is None:
                calibrated_center = (int(face_landmarks.landmark[1].x * img_w),
                                     int(face_landmarks.landmark[1].y * img_h))
  
            
            
            
                    # Deteksi arah kepala berpaling
        head_direction, angle_diff = detect_head_turn(face_landmarks, img_w, img_h)
        cv2.putText(image, f"Kepala: {head_direction}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
    
       #  Gunakan posisi kalibrasi, bukan tengah layar default 
        nose_tip = face_landmarks.landmark[1]
        center_x = int(nose_tip.x * img_w)
        center_y = int(nose_tip.y * img_h)
        radius = 50  # zona aman

            # Titik hidung sekarang
        nose_point = (center_x, center_y)

            # Gambar lingkaran yang mengikuti hidung
        cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
        # cv2.circle(image, nose_point, 5, (0, 0, 255), -1)

            # Jarak gerakan hidung terhadap lingkaran (harus kecil karena lingkaran ikut bergerak)
        dx = nose_point[0] - center_x
        dy = nose_point[1] - center_y
        distance = np.sqrt(dx**2 + dy**2)
            
            

    cv2.imshow('Deteksi Arah Pandang Mata', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()