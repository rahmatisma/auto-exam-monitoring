import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Inisialisasi face mesh dari MediaPipe untuk mendeteksi titik-titik wajah
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

# Variabel global
calibrated_center = None
calibration_frames = 0
CALIBRATION_REQUIRED = 100  # Kalibrasi selama 30 frame

# Untuk mencatat deteksi gerakan mata kiri/kanan
last_direction = None
direction_start_time = None
captured = False

# Buat folder untuk menyimpan bukti jika belum ada
if not os.path.exists("Bukti Menyontek"):
    os.makedirs("Bukti Menyontek")

def get_gaze_direction(left_eye_landmarks, right_eye_landmarks):
    """
    Menghitung arah pandangan berdasarkan posisi iris mata
    """
    # Gaze mata kiri
    left_width = left_eye_landmarks[1][0] - left_eye_landmarks[0][0]
    left_ratio = (left_eye_landmarks[2][0] - left_eye_landmarks[0][0]) / left_width if left_width != 0 else 0.5

    # Gaze mata kanan
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
    """
    Mendeteksi arah kepala berdasarkan posisi hidung relatif terhadap pipi
    """
    nose_tip = face_landmarks.landmark[1]
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]

    nose_x = int(nose_tip.x * img_w)
    left_x = int(left_cheek.x * img_w)
    right_x = int(right_cheek.x * img_w)

    dist_left = abs(nose_x - left_x)
    dist_right = abs(nose_x - right_x)

    ratio = dist_left / dist_right if dist_right != 0 else 1

    # Mirroring arah kepala 
    if ratio > 1.2:
        direction = "KE KANAN" 
    elif ratio < 0.8:
        direction = "KE KIRI"  
    else:
        direction = "TENGAH"

    angle_diff = abs(dist_left - dist_right)
    return direction, angle_diff

def draw_eye_landmarks(image, left_eye_coords, right_eye_coords):
    """
    Menggambar landmark mata dan garis batas
    """
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

def process_gaze_capture(direction, image):
    """
    Memproses capture otomatis saat terdeteksi menyontek
    """
    global last_direction, direction_start_time, captured
    
    current_time = time.time()

    if direction in ["KIRI", "KANAN"]:
        if last_direction != direction:
            last_direction = direction
            direction_start_time = current_time
            captured = False
        else:
            elapsed = current_time - direction_start_time
            if elapsed >= 1.0 and not captured:
                # Ambil capture saat deteksi arah yang sama selama 1 detik
                captured = True
                timestamp = int(time.time())
                filename = f"Bukti Menyontek/capture_{direction.lower()}_{timestamp}.jpg"
                cv2.imwrite(filename, image)
                
                # Tampilkan hasil capture di jendela baru
                captured_img = cv2.imread(filename)
                cv2.imshow(f'Tangkapan Menyontek ke {direction}', captured_img)
                print(f"Bukti tersimpan: {filename}")
    else:
        last_direction = None
        direction_start_time = None
        captured = False

def detect_cheating_by_nose_position(face_landmarks, img_w, img_h, calibrated_center):
    """
    Mendeteksi kecurangan berdasarkan pergerakan hidung dari posisi kalibrasi
    """
    nose_tip = face_landmarks.landmark[1]
    current_nose_x = int(nose_tip.x * img_w)
    current_nose_y = int(nose_tip.y * img_h)
    
    radius = 50  # zona aman
    
    # Hitung jarak dari posisi kalibrasi
    dx = current_nose_x - calibrated_center[0]
    dy = current_nose_y - calibrated_center[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Gambar lingkaran zona aman di posisi kalibrasi
    cv2.circle(image, calibrated_center, radius, (0, 255, 0), 2)
    
    # Gambar titik hidung saat ini
    cv2.circle(image, (current_nose_x, current_nose_y), 5, (0, 0, 255), -1)
    
    # Deteksi kecurangan jika keluar dari zona aman
    cheating_status = "AMAN"
    if distance > radius:
        if abs(dx) > abs(dy):
            cheating_status = "MENYONTEK (Kiri/Kanan)"
        else:
            cheating_status = "MENYONTEK (Atas/Bawah)"
    
    return cheating_status, distance

# Mulai menangkap video
cap = cv2.VideoCapture(0)

print("Sistem Deteksi Arah Pandangan Dimulai")
print("Instruksi:")
print("- Posisikan wajah di tengah layar")
print("- Lihat lurus ke depan selama kalibrasi")
print("- Tekan 'q' untuk keluar")
print("- Tekan 'r' untuk reset kalibrasi")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Gagal membaca dari kamera")
        break

    # Flip horizontal untuk efek mirror
    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            try:
                # === KALIBRASI POSISI AWAL ===
                if calibrated_center is None:
                    calibration_frames += 1
                    nose_tip = face_landmarks.landmark[1]
                    temp_x = int(nose_tip.x * img_w)
                    temp_y = int(nose_tip.y * img_h)
                    
                    # Tampilkan status kalibrasi
                    cv2.putText(image, f"KALIBRASI: {calibration_frames}/{CALIBRATION_REQUIRED}", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(image, "Lihat lurus ke depan!", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Selesai kalibrasi
                    if calibration_frames >= CALIBRATION_REQUIRED:
                        calibrated_center = (temp_x, temp_y)
                        print(f"Kalibrasi selesai: {calibrated_center}")
                    
                    continue

                # === DETEKSI ARAH PANDANGAN MATA ===
                # Landmark mata kiri: sudut kiri (33), sudut kanan (133), iris (468)
                left_eye_lm = [face_landmarks.landmark[33], face_landmarks.landmark[133], face_landmarks.landmark[468]]
                # Landmark mata kanan: sudut kanan (362), sudut kiri (263), iris (473)
                right_eye_lm = [face_landmarks.landmark[362], face_landmarks.landmark[263], face_landmarks.landmark[473]]

                # Konversi ke koordinat piksel
                left_eye_coords = [(int(p.x * img_w), int(p.y * img_h)) for p in left_eye_lm]
                right_eye_coords = [(int(p.x * img_w), int(p.y * img_h)) for p in right_eye_lm]

                # Hitung arah pandangan
                gaze_direction, gaze_ratio = get_gaze_direction(left_eye_coords, right_eye_coords)

                # === DETEKSI ARAH KEPALA ===
                head_direction, angle_diff = detect_head_turn(face_landmarks, img_w, img_h)

                # === DETEKSI KECURANGAN BERDASARKAN POSISI HIDUNG ===
                nose_cheating_status, nose_distance = detect_cheating_by_nose_position(
                    face_landmarks, img_w, img_h, calibrated_center)

                # === PROSES CAPTURE OTOMATIS ===
                process_gaze_capture(gaze_direction, image)

                # === VISUALISASI ===
                draw_eye_landmarks(image, left_eye_coords, right_eye_coords)

                # Tampilkan informasi
                cv2.putText(image, f"Mata: {gaze_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f"Rasio: {gaze_ratio:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f"Kepala: {head_direction}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(image, f"Status: {nose_cheating_status}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f"Jarak: {nose_distance:.1f}px", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Tampilkan intensitas kecurangan dengan warna
                if "MENYONTEK" in nose_cheating_status:
                    cv2.putText(image, "PERINGATAN!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            except IndexError as e:
                print(f"Error dalam deteksi landmark: {e}")
                continue

    else:
        # Tidak ada wajah terdeteksi
        cv2.putText(image, "Tidak ada wajah terdeteksi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame
    cv2.imshow('Sistem Deteksi Arah Pandangan', image)

    # Kontrol keyboard
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset kalibrasi
        calibrated_center = None
        calibration_frames = 0
        print("Kalibrasi direset")

print("Sistem dihentikan")
cap.release()
cv2.destroyAllWindows()