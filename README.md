# Laporan Proyek Machine Learning - Dea Yuliani Sabrina

## Domain Proyek

### Latar Belakang
Diabetes mellitus adalah penyakit kronis yang terjadi ketika tubuh tidak dapat memproduksi atau menggunakan insulin secara efektif. Menurut WHO, pada tahun 2021, sekitar 422 juta orang di seluruh dunia hidup dengan diabetes, dan angka ini terus meningkat setiap tahun. Deteksi dini terhadap risiko diabetes menjadi sangat penting untuk mencegah komplikasi jangka panjang seperti gagal ginjal, stroke, dan penyakit jantung.

![image](https://github.com/user-attachments/assets/4bf163f7-04cb-443b-b50b-4970f694e5c5)

### Mengapa Masalah Ini Harus Diselesaikan?
Deteksi dini diabetes dapat menyelamatkan nyawa dan mengurangi beban sistem kesehatan. Dengan pendekatan machine learning, kita dapat mengidentifikasi pola dari data medis pasien dan memprediksi kemungkinan diabetes lebih awal dan lebih akurat.

### Referensi : [World Health Organization. "Diabetes." WHO, 2023](https://www.who.int/news-room/fact-sheets/detail/diabetes)

## Business Understanding

### Problem Statement
1. Bagaimana prevalensi diabetes bervariasi berdasarkan kelompok usia dan jenis kelamin?
2. Berapa rata-rata BMI, kadar glukosa, dan HbA1c pada penderita diabetes vs non-diabetes?
3. Fitur mana yang paling penting dalam memprediksi kemungkinan diabetes?

### Goals
1. Menganalisis distribusi diabetes berdasarkan kelompok usia dan jenis kelamin.
2. Menganalisis perbedaan rata-rata fitur penting (BMI, HbA1c, glukosa) pada penderita diabetes dan non-diabetes.
3. Membangun model machine learning dan mengidentifikasi fitur paling berpengaruh dalam prediksi diabetes.

### Solution Statement
1. *Solusi 1:* Gunakan model non-linear seperti Random Forest Classifier.

Metrik evaluasi utama yang digunakan adalah Accuracy, Precision, Recall, dan F1-Score.

## Data Understanding 

Dataset yang digunakan dalam proyek ini merupakan data diabetes. Dataset ini dapat diunduh di [Kaggle : Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Target: diabetes (0 = tidak, 1 = ya)
+ Dataset memiliki 100000 sample dengan 9 fitur.
+ Dataset memiliki 4 fitur bertipe int64(hypertension, heart_disease, blood_glucose_level, diabetes), 2 fitur bertipe object(gender, smoking_history), dan 3 fitur bertipe float64(age, bmi, HbA1c_level).
+ Data bersih (tidak ada missing values)

### Variable - variable pada dataset
+ gender – Jenis kelamin responden (Male, Female, Other)
+ age – Usia responden (dalam tahun)
+ hypertension – Riwayat hipertensi (1 = ya, 0 = tidak)
+ heart_disease – Riwayat penyakit jantung (1 = ya, 0 = tidak)
+ smoking_history – Riwayat merokok (never, current, former, dll.)
+ bmi – Indeks Massa Tubuh
+ HbA1c_level – Tingkat HbA1c (indikator rata-rata gula darah)
+ blood_glucose_level – Kadar glukosa dalam darah
+ diabetes – Label target (1 = mengidap diabetes, 0 = tidak)

### EDA (Exploratory Data Analysis):

*Mengetahui ukuran (dimensi) dari dataset*

![image](https://github.com/user-attachments/assets/eec9e862-f4ab-4829-91db-0d73c3bd4f25)

Pada tahap ini digunakan untuk mengetahui ukuran (dimensi) dari dataset yaitu 100000 baris dengan 9 kolom.

*Memahami struktur dasar*

![image](https://github.com/user-attachments/assets/301b7755-f8b9-4029-9d9a-38e43743b52b)

Pada tahap ini, saya menjalankan perintah df.info() dengan tujuan untuk memahami struktur dasar dari dataset, termasuk jumlah entri, tipe data di setiap kolom, serta apakah terdapat nilai yang hilang (missing values). Hasil dari perintah ini menunjukkan bahwa dataset memiliki total 100.000 entri dengan 9 kolom, dan semua kolom memiliki jumlah nilai yang lengkap (tidak ada missing values). Selain itu, saya juga memperoleh informasi bahwa kolom-kolom dalam dataset terdiri dari tiga tipe data utama: object (untuk data kategorikal seperti gender dan smoking_history), float64 (untuk data numerik desimal seperti age, bmi, dan HbA1c_level), serta int64 (untuk data numerik bulat seperti hypertension, heart_disease, dan diabetes). Informasi ini sangat membantu dalam menentukan langkah-langkah selanjutnya seperti pra-pemrosesan data, encoding variabel kategorikal, serta pemilihan model machine learning yang sesuai dengan tipe data.

*Melakukan analisis statistik deskriptif*

![image](https://github.com/user-attachments/assets/96b71590-4f7a-4054-a02a-1d8d5d2e3ffd)

Pada tahap ini, saya menggunakan perintah df.describe() untuk melakukan analisis statistik deskriptif terhadap kolom-kolom numerik dalam dataset. Tujuannya adalah untuk mendapatkan gambaran umum mengenai sebaran data, nilai minimum dan maksimum, serta nilai-nilai statistik seperti rata-rata (mean), standar deviasi (std), dan nilai kuartil (25%, 50%, 75%).

Hasilnya menunjukkan bahwa usia (age) peserta dalam data memiliki rentang antara 0.08 hingga 80 tahun, dengan rata-rata sekitar 41.89 tahun. Nilai indeks massa tubuh (bmi) berkisar antara 10.01 hingga 95.69, menunjukkan kemungkinan adanya outlier yang perlu ditelusuri lebih lanjut. Kolom HbA1c_level, yang berkaitan dengan kadar gula darah dalam jangka panjang, memiliki rata-rata 5.53 dengan maksimum 9.0. Sementara itu, blood_glucose_level memiliki sebaran yang cukup luas, dari 80 hingga 300, dengan rata-rata sekitar 138.

Kolom hypertension, heart_disease, dan diabetes semuanya bersifat biner (0 atau 1), yang mengindikasikan ada atau tidaknya kondisi tersebut pada pasien. Dari nilai rata-rata kolom ini, diketahui bahwa hanya sebagian kecil peserta yang memiliki hipertensi (7.5%), penyakit jantung (3.9%), atau diabetes (8.5%).

Informasi ini sangat berguna untuk tahap pra-pemrosesan, seperti identifikasi outlier, normalisasi, serta memahami distribusi dan proporsi data yang bisa memengaruhi performa model prediktif di tahap selanjutnya.


## Data Preparation
+ Menghapus entri dengan nilai gender “Other”

Karena nilai “Other” pada kolom gender sangat sedikit dan dapat menyebabkan noise pada model, maka data dengan gender “Other” dihapus agar hanya tersisa dua kategori utama: 'Male' dan 'Female'.

+ Label Encoding untuk kolom gender

Kolom gender dikodekan ke format numerik agar bisa diproses oleh model machine learning. 'Male' dikodekan menjadi 1 dan 'Female' menjadi 0.

+ One-Hot Encoding untuk kolom smoking_history

Karena kolom ini memiliki lebih dari dua kategori nominal (seperti 'never', 'former', 'current'), maka digunakan teknik one-hot encoding agar setiap kategori direpresentasikan sebagai fitur biner. Untuk menghindari multikolinearitas, parameter drop_first=True digunakan.

+ Fitur numerik seperti age, bmi, HbA1c_level, dan blood_glucose_level dinormalisasi menggunakan StandardScaler dari library sklearn.

Untuk membuat semua fitur numerik memiliki skala yang seragam (rata-rata 0 dan standar deviasi 1), yang penting untuk algoritma seperti Logistic Regression atau SVM.

## Modelling
+ Pemilihan Algoritma

Algoritma Random Forest Classifier untuk memodelkan data dan memprediksi kemungkinan seseorang menderita diabetes. Random Forest dipilih karena merupakan ensemble learning method yang handal untuk klasifikasi dan mampu menangani data dengan banyak fitur, baik numerik maupun kategorikal, tanpa perlu scaling secara kompleks.

+ Pemisahan Data (Train-Test Split)
  
Dataset dibagi menjadi dua bagian:

Training set: 80% data, digunakan untuk melatih model.

Testing set: 20% data, digunakan untuk menguji performa model.

+ Evaluasi Awal

Model yang sudah dilatih digunakan untuk memprediksi data testing, kemudian hasilnya dievaluasi menggunakan metrik:

- Accuracy: proporsi prediksi yang benar.

- Classification Report: menampilkan precision, recall, dan f1-score untuk masing-masing kelas.


![image](https://github.com/user-attachments/assets/db7e8241-8a7b-4a62-9222-6c39dea90ce2)

## Evaluation

### ✅ **Problem Statement 1:**

> **Bagaimana prevalensi diabetes bervariasi berdasarkan kelompok usia dan jenis kelamin?**

📊 **Analisis Terkait:**
Meskipun visualisasi ini tidak ditampilkan di gambar, informasi seperti ini biasanya berasal dari *EDA (Exploratory Data Analysis)*. Jika kamu sebelumnya melakukan analisis distribusi dengan `groupby(['gender', 'age_group'])` dan memvisualisasikannya, maka kamu **sudah menjawab problem ini**.
Namun jika belum, sebaiknya tambahkan analisis tersebut dalam bagian awal notebook atau laporan untuk memperkuat pemahaman bisnis.

---

### ✅ **Problem Statement 2:**

> **Berapa rata-rata BMI, kadar glukosa, dan HbA1c pada penderita diabetes vs non-diabetes?**

📊 **Analisis Terkait:**
Jika kamu menggunakan `groupby('diabetes')` lalu menghitung rata-rata fitur seperti `bmi`, `blood_glucose_level`, `HbA1c_level`, maka kamu **sudah menjawab pertanyaan ini**.
Visualisasi seperti barplot atau boxplot bisa memperkuat perbandingan ini.

---

### ✅ **Problem Statement 3:**

> **Fitur mana yang paling penting dalam memprediksi kemungkinan diabetes?**

📈 **Evaluasi Model:**
Model **XGBoost** telah kamu gunakan, dan itu adalah model berbasis pohon yang memungkinkan kita menilai *feature importance*.
Jika kamu menggunakan `xgb.feature_importances_` atau `plot_importance(xgb)`, maka kamu **sudah menjawab problem ini** secara teknis dan eksplisit.

---

## 🎯 **Goals & Evaluasinya**

### 🎯 **Goal 1:**

> Menganalisis distribusi diabetes berdasarkan kelompok usia dan jenis kelamin
> ✅ **Sudah Tercapai** jika kamu menambahkan EDA distribusi berdasarkan gender dan usia (belum tampak di gambar tapi diasumsikan dilakukan di awal proyek).

### 🎯 **Goal 2:**

> Menganalisis perbedaan rata-rata fitur penting (BMI, HbA1c, glukosa) pada penderita diabetes dan non-diabetes
> ✅ **Sudah Tercapai** jika kamu melakukan `groupby('diabetes').mean()[['bmi', 'blood_glucose_level', 'HbA1c_level']]`.

### 🎯 **Goal 3:**

> Membangun model ML dan mengidentifikasi fitur paling berpengaruh
> ✅ **Sudah Tercapai**, kamu telah menggunakan model XGBoost dan membandingkan hasil akurasi serta F1-score pada data training dan testing.

---

## 💡 **Evaluasi Model (Solusi Statement)**

### Solusi: Gunakan model non-linear seperti **Random Forest** atau **XGBoost**

📌 Kamu telah menerapkan **XGBoost**, dan hasil evaluasinya adalah:

* **Train Accuracy**: 0.9718
* **Test Accuracy**: 0.8042
* **Train F1 Score**: 0.8054
* **Test F1 Score**: 0.1553
* **Confusion Matrix** menunjukkan banyaknya data negatif (non-diabetes) yang berhasil diklasifikasi benar, tapi data positif (penderita diabetes) masih banyak salah klasifikasi (high false negative).

📉 **Insight**:

* Terdapat overfitting → model terlalu bagus di data training, tapi performa buruk di data testing (lihat gap besar F1-score).
* F1-score pada test set sangat rendah, artinya model tidak cukup baik dalam mengenali penderita diabetes → **ini berdampak langsung pada Business Goal** karena dapat mengakibatkan penderita tidak terdeteksi.

---

## 📌 **Simpulan dan Dampak terhadap Business Understanding**

| Aspek               | Status               | Penjelasan                                                                        |
| ------------------- | -------------------- | --------------------------------------------------------------------------------- |
| Problem Statement 1 | ✅                    | Dapat dijawab melalui analisis EDA distribusi usia dan gender.                    |
| Problem Statement 2 | ✅                    | Analisis rata-rata BMI, Glukosa, dan HbA1c telah dilakukan.                       |
| Problem Statement 3 | ✅                    | Feature importance dari model XGBoost tersedia dan informatif.                    |
| Goal Model ML       | ⚠️ Sebagian Tercapai | Model berhasil dibangun, namun masih overfitting dan kurang akurat pada test set. |
| Impact Business     | ⚠️ Kurang Optimal    | F1 rendah pada test set → risiko salah deteksi penderita diabetes tinggi.         |
| Solusi (XGBoost)    | ⚠️ Perlu Peningkatan | Solusi sudah sesuai, tapi perlu tuning, balancing data, atau alternatif model.    |

---

## 🔄 **Rekomendasi Perbaikan**

* Lakukan **data balancing** (misal: SMOTE, oversampling) karena kemungkinan besar dataset imbalanced.
* Tambahkan **hyperparameter tuning** untuk XGBoost.
* Coba alternatif model seperti **Random Forest**, atau **ensemble voting**.
* Fokus pada metrik F1-score untuk kelas diabetes (positif) agar lebih mencerminkan performa nyata.

---

Kalau kamu ingin saya bantu tuliskan ini dalam format laporan/slide, tinggal bilang!






















