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

**Penjelasan Kode**

Di bagian ini, saya membuat visualisasi untuk membandingkan performa tiga model machine learning (Logistic Regression, Random Forest, dan XGBoost) berdasarkan metrik **accuracy** dan **F1-score**, baik di data pelatihan (**train**) maupun data pengujian (**test**). Langkah-langkahnya sebagai berikut:

1. **Mengambil data dari DataFrame `models`**:

   * `acc_df` memuat data akurasi untuk training dan testing.
   * `f1_df` memuat data F1-score untuk training dan testing.

2. **Membuat dua grafik berdampingan** (`fig, axes = plt.subplots(1, 2)`):

   * Grafik kiri menampilkan **akurasi** (train dan test) dari masing-masing model.
   * Grafik kanan menampilkan **F1-score** (train dan test).

3. **Pengaturan visualisasi**:

   * Skala sumbu y dibatasi dari 0 sampai 1 untuk konsistensi.
   * Warna dan legenda ditentukan agar perbandingan antar metrik lebih jelas.
   * `plt.tight_layout()` digunakan agar grafik tidak saling menumpuk.

**Penjelasan Grafik**

Di tahap ini, saya membandingkan performa tiga model — **Logistic Regression**, **Random Forest**, dan **XGBoost** — dengan menggunakan metrik akurasi dan F1-score. Dari grafik yang ditampilkan, kita bisa menyimpulkan hal berikut:

🔹 Logistic Regression

* **Akurasi pelatihan** sangat tinggi (\~97%), namun **akurasi pengujian** hanya sekitar 8,4%.
* **F1-score pelatihan** cukup tinggi (\~73%), tapi **F1-score pengujian** hanya sekitar 15,5%.
* Ini menunjukkan model ini **belajar terlalu banyak dari data pelatihan** (overfitting), dan gagal mengenali pola di data baru.

🔹 Random Forest

* Sama seperti Logistic Regression, performa di data pelatihan sangat baik (akurasi \~98%, F1-score \~79%), namun turun drastis di data uji.
* Hal ini mengindikasikan bahwa model ini juga mengalami **overfitting**.

🔹 XGBoost

* Di tahap ini, saya mengevaluasi model XGBoost dengan menggunakan fungsi `evaluate_model`.
  Hasil evaluasi menunjukkan bahwa model ini memiliki performa sangat bagus di data latih, dengan akurasi sekitar **97,18%** dan F1-score sebesar **80,54%**. Ini artinya model berhasil mempelajari pola-pola dari data latih dengan sangat baik.

* Namun, saat diuji pada data uji, **akurasi dan F1-score turun drastis** menjadi masing-masing sekitar **8,42%** dan **15,53%**, sama seperti model lain sebelumnya. Hal ini menunjukkan bahwa model XGBoost juga kesulitan untuk menggeneralisasi ke data baru dan **mungkin mengalami overfitting**.

**Business Understanding dan Evaluasi Model**

**1. Apakah sudah menjawab setiap Problem Statement?**

**✅ Bagaimana prevalensi diabetes bervariasi berdasarkan kelompok usia dan jenis kelamin?**
Ya. Pertanyaan ini dapat dijawab dengan melakukan analisis eksploratif data (EDA) menggunakan visualisasi distribusi berdasarkan kolom usia dan jenis kelamin. Hal ini memberi insight penting bagi pihak medis atau pemerintah untuk memahami segmen populasi berisiko tinggi.

**✅ Berapa rata-rata BMI, kadar glukosa, dan HbA1c pada penderita diabetes vs non-diabetes?**
Sudah dijawab melalui perbandingan nilai rata-rata tiap fitur tersebut pada dua kelompok (diabetes dan non-diabetes). Ini memberikan gambaran jelas tentang indikator kesehatan yang paling membedakan kedua kelompok.

**✅ Fitur mana yang paling penting dalam memprediksi kemungkinan diabetes?**
Telah dijawab dengan membangun model machine learning dan menganalisis feature importance, terutama pada model seperti Random Forest atau XGBoost. Fitur seperti HbA1c, Glukosa, dan BMI terbukti menjadi indikator utama.

**2. Apakah berhasil mencapai setiap Goals?**

| Goals                                                               | Status | Penjelasan Singkat                                                    |
| ------------------------------------------------------------------- | ------ | --------------------------------------------------------------------- |
| Menganalisis distribusi diabetes berdasarkan kelompok usia & gender | ✅      | Sudah dianalisis melalui visualisasi dan statistik deskriptif         |
| Menganalisis perbedaan rata-rata BMI, HbA1c, Glukosa                | ✅      | Sudah dibedakan antara kelompok diabetes dan non-diabetes             |
| Membangun model ML & identifikasi fitur penting                     | ✅      | Model dibangun & fitur penting sudah ditentukan (HbA1c, Glukosa, BMI) |

---

**3. Evaluasi Solusi yang Diberikan (Solution Statement)**

**Solusi: Gunakan model non-linear seperti Random Forest dan XGBoost.**

| Model              | Train Accuracy | Test Accuracy | Train F1 | Test F1 | Evaluasi    |
| ------------------ | -------------- | ------------- | -------- | ------- | ----------- |
| LogisticRegression | 95.95%         | 8.42%         | 72.37%   | 15.52%  | Overfitting |
| RandomForest       | 97.13%         | 8.42%         | 79.93%   | 15.52%  | Overfitting |
| XGBoost            | 97.18%         | 8.42%         | 80.53%   | 15.52%  | Overfitting |

**🧩 Kesimpulan:**

* Semua model, meskipun sangat akurat pada data pelatihan, **gagal generalisasi** ke data pengujian.
* Ini mengindikasikan **overfitting parah** — model terlalu menghafal data latih dan tidak mengenali pola baru.
* Solusi yang direncanakan **masih belum memberikan dampak optimal terhadap prediksi**.

---

**Rekomendasi Lanjutan**

Untuk memastikan solusi benar-benar berdampak terhadap bisnis dan menjawab kebutuhan nyata, berikut saran lanjutan:

1. **Cross-validation** untuk validasi yang lebih stabil dan adil.
2. **Penyeimbangan data** jika terdapat ketimpangan jumlah kelas (SMOTE, oversampling, dsb).
3. **Feature selection & regularisasi** untuk mengurangi kompleksitas dan overfitting.
4. Tambahkan **metrik bisnis** seperti jumlah pasien benar teridentifikasi, potensi penghematan biaya kesehatan, dll.

---

**Kesimpulan Akhir**

* Semua problem statement dan goals **sudah terjawab secara teknis**.
* Namun, model yang dibangun **belum efektif digunakan dalam konteks nyata** karena performanya buruk pada data uji.
* Solusi yang direncanakan **masih perlu penguatan** dengan pendekatan yang lebih robust agar berdampak langsung pada pengambilan keputusan dalam manajemen kesehatan dan penanganan diabetes.



















