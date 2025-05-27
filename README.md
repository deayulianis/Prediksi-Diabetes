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

*Pertanyaan 1: Prevalensi berdasarkan usia dan gender*

![image](https://github.com/user-attachments/assets/8ce8b633-d0bc-4226-8499-1c70e614c8e4)

Berdasarkan output tersebut, prevalensi diabetes dalam kelompok usia <30 tahun bervariasi berdasarkan jenis kelamin sebagai berikut:

+ Laki-laki (gender = 0):

Tidak diabetes (label 0): 88.92%

Diabetes (label 1): 13.08%

+ Perempuan (gender = 1):

Tidak diabetes: 82.83%

Diabetes: 17.17%

*Pertanyaan 2: Rata-rata BMI, Glukosa, dan HbA1c*

![image](https://github.com/user-attachments/assets/b7be0b66-906b-49a4-8d0c-195a7c8481ed)

Berdasarkan ouput yang diberikan, berikut adalah rata-rata BMI, kadar glukosa darah, dan kadar HbA1c pada penderita diabetes dibandingkan dengan non-diabetes:

+ Non-Diabetes (diabetes = 0):
  
BMI: -0.0653

Kadar glukosa darah: -0.1279

HbA1c: -0.1221

+ Penderita Diabetes (diabetes = 1):
  
BMI: 0.7033

Kadar glukosa darah: 1.3765

HbA1c: 1.3145

*Pertanyaan 3: Fitur terpenting*

![image](https://github.com/user-attachments/assets/b1f559f1-8dc1-4aab-92e5-b501e83a5d76)

*Metrik Evaluasi yang Digunakan:*

Model dievaluasi menggunakan beberapa metrik klasifikasi berikut yang disediakan oleh sklearn.metrics:

+ Accuracy : Mengukur proporsi prediksi yang benar terhadap total jumlah prediksi.

![image](https://github.com/user-attachments/assets/1fea8e1a-cc54-4bf3-b37c-6f3288001032)

+ Precision : Mengukur proporsi prediksi positif yang benar.

+ Recall (Sensitivity) : Mengukur proporsi kasus positif yang berhasil diprediksi dengan benar.

+ F1-Score : Merupakan harmonisasi dari precision dan recall, digunakan saat distribusi kelas tidak seimbang.

+ Feature Importance :  Untuk menilai kontribusi setiap fitur terhadap model Random Forest.

*Hasil Evaluasi Model:*

+ Accuracy:
  
Model mencapai akurasi sebesar 96.97% pada data uji. Ini menunjukkan bahwa model cukup handal dalam memprediksi apakah seseorang menderita diabetes atau tidak berdasarkan fitur-fitur input.

![image](https://github.com/user-attachments/assets/8eb8fde7-d313-48cc-bf94-b57db489e32d)

+ Classification Report:

![image](https://github.com/user-attachments/assets/ba1837b8-e523-474c-8447-29a75d342a6f)

*Analisis:*

+ Model memiliki precision dan F1-score yang tinggi untuk kelas Non-Diabetic, namun performa untuk kelas Diabetic memiliki recall yang lebih rendah (0.69), menunjukkan bahwa masih ada sejumlah penderita diabetes yang tidak teridentifikasi oleh model.

+ Ini menunjukkan potensi ketidakseimbangan kelas (jumlah penderita diabetes jauh lebih sedikit), sehingga pendekatan seperti class weighting atau oversampling bisa dijadikan pertimbangan perbaikan ke depan.

*Feature Importance:*

![image](https://github.com/user-attachments/assets/22192d47-784f-4862-9ebc-1df2101d221f)

*Fitur paling penting berdasarkan nilai feature importance:*
+ blood_glucose_level
+ HbA1c_level
+ bmi
+ age

Fitur-fitur ini relevan secara klinis dan konsisten dengan studi medis terkait deteksi diabetes.





















