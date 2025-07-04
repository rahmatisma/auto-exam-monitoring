# 📌 Deteksi Arah Pandang dan Kepala dengan MediaPipe

Sistem ini menggunakan **MediaPipe Face Mesh** untuk mendeteksi arah pandang mata (gaze tracking) dan arah kepala (head pose estimation) secara real-time melalui kamera webcam. Sistem ini juga dapat mendeteksi indikasi "menyontek" berdasarkan posisi pandangan dan pergerakan kepala terhadap zona aman.

---

## 🎯 Fitur

- Deteksi arah pandangan mata: **KIRI**, **KANAN**, atau **TENGAH**.
- Estimasi arah kepala: **KE KIRI**, **KE KANAN**, atau **TENGAH**.
- Visualisasi landmark mata dan hidung.
- Sistem peringatan jika arah pandang keluar dari zona aman (indikasi kecurangan).
- Kalibrasi otomatis saat pertama kali dijalankan.

---

## 🧠 Metodologi Pengenalan

### 1. Deteksi Landmark Wajah

Menggunakan `mediapipe.solutions.face_mesh` untuk mendeteksi 468 titik (landmark) pada wajah, termasuk mata, hidung, dan pipi. Titik-titik penting yang digunakan:

- **Mata kiri**: `33` (ujung kiri), `133` (ujung kanan), `468` (iris tengah)
- **Mata kanan**: `263` (ujung kiri), `362` (ujung kanan), `473` (iris tengah)
- **Hidung**: `1` (ujung hidung)
- **Pipi kiri & kanan**: `234` dan `454`

### 2. Deteksi Arah Pandangan Mata (Gaze Tracking)

Arah pandangan ditentukan berdasarkan posisi **iris** relatif terhadap **ujung mata kiri dan kanan**.

```python
ratio = (iris_x - eye_left_corner_x) / (eye_right_corner_x - eye_left_corner_x)
```

Nilai rata-rata dari mata kiri dan kanan dibandingkan:

- < 0.45: **KIRI**
- > 0.55: **KANAN**
- Di antara: **TENGAH**

### 3. Deteksi Arah Kepala (Head Pose Estimation)

Menggunakan jarak antara hidung dan kedua pipi untuk memperkirakan kemiringan kepala:

```python
ratio = dist_left / dist_right
```

- Jika hidung lebih dekat ke pipi kiri → kepala menghadap kanan
- Jika hidung lebih dekat ke pipi kanan → kepala menghadap kiri
- Jika seimbang → kepala lurus (TENGAH)

### 4. Zona Aman & Deteksi Kecurangan

- Saat wajah pertama kali dikenali, sistem menyimpan posisi kalibrasi (`center_x`, `center_y`) berdasarkan titik hidung.
- Radius zona aman digambarkan sebagai lingkaran hijau.
- Jika titik pandangan (titik merah) keluar dari zona ini, maka sistem menampilkan peringatan seperti:

  - **"Menyontek (Kiri/Kanan)"**
  - **"Menyontek (Atas/Bawah)"**

---

## ▶️ Cara Menjalankan

### ✅ Prasyarat

- Python 3.7+
- Webcam aktif
- Modul Python:
  - `opencv-python`
  - `mediapipe`
  - `numpy`

### 💻 Instalasi

```bash
pip install opencv-python mediapipe numpy
```

### 🚀 Menjalankan Program

```bash
python deteksi_pandangan.py
```

Tekan `q` untuk keluar dari aplikasi.

---

## 📸 Tampilan Visual

- Titik **kuning/magenta** menunjukkan landmark mata.
- Titik **merah** menunjukkan arah pandang.
- Lingkaran **hijau** adalah zona aman dari arah hidung.
- Status seperti **"Kepala: KE KIRI"** atau **"Menyontek"** akan muncul jika terjadi pelanggaran arah pandang.

---

## 📂 Struktur Proyek

```
📁 proyek-pandangan/
├── deteksi_pandangan.py
└── README.md
```

---

## 📄 Lisensi

Proyek ini bersifat open-source dan bebas digunakan untuk keperluan pembelajaran, penelitian, atau pengembangan lebih lanjut.
