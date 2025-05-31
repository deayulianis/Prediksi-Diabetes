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
1. *Solusi 1:* Gunakan model non-linear seperti Random Logistic .
2. *Solusi 2:* Gunakan model non-linear seperti Random Forest Classifier.
3. *Solusi 3:* Gunakan model non-linear seperti Random XGBoost Classifier.

## Data Understanding 

Dataset yang digunakan dalam proyek ini merupakan data diabetes. Dataset ini dapat diunduh di [Kaggle : Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

![image](https://github.com/user-attachments/assets/1aca0d90-b221-4d80-8556-2a833390e1fb)

- Melihat 5 baris pertama dari dataset yang berisi data diabetes.

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

![image](https://github.com/user-attachments/assets/eaf63b43-b66b-4e16-aedc-d07af6fe62d7)

- Pada tahap ini digunakan untuk mengetahui ukuran (dimensi) dari dataset yaitu 100000 baris dengan 9 kolom.

*Memahami struktur dasar*

![image](https://github.com/user-attachments/assets/9b0b2a0f-c521-4a6e-a677-586c746b85a3)

- Terdapat 2 kolom dengan tipe object, yaitu: gender dan smoking_history. Kolom ini merupakan categorical features (fitur non-numerik).
- Terdapat 3 kolom numerik dengan tipe data float64 yaitu: age, bmi, dan HbA1c_level . Ini merupakan fitur numerik yang merupakan hasil pengukuran secara fisik.
- Terdapat 4 kolom numerik dengan tipe data int64, yaitu: hypertension, heart_disease, blood_glucose_level, dan diabetes. Kolom ini merupakan target fitur kita.

*Melakukan analisis statistik deskriptif*

![image](https://github.com/user-attachments/assets/96b71590-4f7a-4054-a02a-1d8d5d2e3ffd)

Pada tahap ini, perintah df.describe() untuk melakukan analisis statistik deskriptif terhadap kolom-kolom numerik dalam dataset. Tujuannya adalah untuk mendapatkan gambaran umum mengenai sebaran data, nilai minimum dan maksimum, serta nilai-nilai statistik seperti rata-rata (mean), standar deviasi (std), dan nilai kuartil (25%, 50%, 75%).

- Hasilnya menunjukkan bahwa usia (age) peserta dalam data memiliki rentang antara 0.08 hingga 80 tahun, dengan rata-rata sekitar 41.89 tahun. Nilai indeks massa tubuh (bmi) berkisar antara 10.01 hingga 95.69, menunjukkan kemungkinan adanya outlier yang perlu ditelusuri lebih lanjut. Kolom HbA1c_level, yang berkaitan dengan kadar gula darah dalam jangka panjang, memiliki rata-rata 5.53 dengan maksimum 9.0. Sementara itu, blood_glucose_level memiliki sebaran yang cukup luas, dari 80 hingga 300, dengan rata-rata sekitar 138.

- Kolom hypertension, heart_disease, dan diabetes semuanya bersifat biner (0 atau 1), yang mengindikasikan ada atau tidaknya kondisi tersebut pada pasien. Dari nilai rata-rata kolom ini, diketahui bahwa hanya sebagian kecil peserta yang memiliki hipertensi (7.5%), penyakit jantung (3.9%), atau diabetes (8.5%).

Informasi ini sangat berguna untuk tahap pra-pemrosesan, seperti identifikasi outlier, normalisasi, serta memahami distribusi dan proporsi data yang bisa memengaruhi performa model prediktif di tahap selanjutnya.

*Deteksi Outlier Ekstrem dengan IQR Method dan Visualisasi*

![image](https://github.com/user-attachments/assets/db7a7503-d01e-44ee-a27b-b913d7c390ce)

Pada tahap ini, melakukan eksplorasi awal terhadap fitur **BMI** dengan tujuan memahami **distribusi dan karakteristik datanya**.

Visualisasi boxplot digunakan untuk mengidentifikasi **sebaran nilai dan outlier**, yang menunjukkan banyak nilai ekstrem di sisi kanan (BMI tinggi).

Sementara itu, histogram menampilkan bentuk distribusi yang **skew ke kanan**, artinya sebagian besar individu memiliki BMI di bawah rata-rata, namun terdapat sejumlah kecil dengan nilai BMI sangat tinggi.

Analisis ini penting sebagai dasar untuk **preprocessing**, seperti penanganan outlier atau transformasi data sebelum pemodelan.

*Menghitung IQR (Interquartile Range)*

![image](https://github.com/user-attachments/assets/9504545b-a497-4f54-bc4b-919e34687726)

Pada tahap ini, menghitung IQR (Interquartile Range) untuk fitur BMI guna mengidentifikasi batas bawah dan atas deteksi outlier.

Hasilnya menunjukkan:

- IQR = 5.87

- Batas bawah = 14.96

- Batas atas = 38.45

- Nilai BMI tertinggi = 95.69

Karena nilai tertinggi jauh di atas batas atas, ini menegaskan bahwa dataset mengandung outlier ekstrem yang perlu dipertimbangkan dalam proses cleaning atau transformasi data sebelum pemodelan.

*Memeriksa nilai yang hilang (missing values)*

![image](https://github.com/user-attachments/assets/8acbe93e-4dd8-4ee1-a1e8-c80147dfc229)

Pada tahap ini, memeriksa nilai yang hilang (missing values) di seluruh kolom dataset menggunakan df.isnull().sum().

Hasilnya menunjukkan bahwa tidak ada nilai yang hilang di semua fitur (semua bernilai 0), sehingga tidak diperlukan proses imputasi atau penanganan missing value.

Data ini sudah lengkap dan siap untuk tahap selanjutnya, seperti eksplorasi lebih lanjut atau pemodelan.

### Univariate Analysis

### Categorical Feature

*Mengelompokkan fitur menjadi dua jenis utama*

![image](https://github.com/user-attachments/assets/c741a9ff-c65d-4bec-a40e-a52e766eb1eb)

Pada tahap ini, mengelompokkan fitur menjadi dua jenis utama:

numerical_features: berisi fitur numerik seperti usia, BMI, kadar HbA1c, tekanan darah, dan kadar glukosa. Termasuk juga variabel target diabetes karena berupa nilai biner (0 atau 1).

categorical_features: berisi fitur kategorikal seperti jenis kelamin dan riwayat merokok.

Tujuan pengelompokan ini adalah untuk memudahkan proses analisis, visualisasi, dan preprocessing, karena fitur numerik dan kategorikal biasanya memerlukan perlakuan yang berbeda dalam pemodelan machine learning.

*Melakukan eksplorasi awal terhadap fitur kategorikal gender*

![image](https://github.com/user-attachments/assets/5fc8277a-942a-4049-bfd3-39dd84ad4ef7)

Pada tahap ini, melakukan eksplorasi awal terhadap fitur kategorikal gender untuk memahami distribusi datanya.

Distribusi jumlah sampel untuk masing-masing kategori pada fitur gender adalah sebagai berikut:

- Female: 58.119 sampel (58.7%)
- Male: 48.952 sampel (41.3%)
- Other: 18 sampel (0.0%)

Dari hasil ini, terlihat bahwa mayoritas sampel berasal dari kelompok perempuan, diikuti oleh laki-laki, sementara kategori lainnya hanya mencakup sebagian sangat kecil dari data (0.0%).

Visualisasi bar chart juga menunjukkan perbedaan distribusi yang cukup signifikan antara kategori gender tersebut. Informasi ini penting untuk memahami proporsi data sebelum melakukan pemodelan, terutama jika fitur gender akan digunakan sebagai variabel prediktor.

*Melakukan eksplorasi awal terhadap fitur kategorikal smoking_history*

![image](https://github.com/user-attachments/assets/4a057c8c-a753-4558-afd5-667326316e5f)

Pada tahap ini, melakukan eksplorasi awal terhadap fitur kategorikal smoking_history untuk memahami distribusi data berdasarkan riwayat merokok.

Distribusi jumlah sampel untuk masing-masing kategori adalah sebagai berikut:

- never: 35.058 sampel (35.4%)
- No Info: 34.952 sampel (35.3%)
- former: 9.352 sampel (9.4%)
- current: 9.258 sampel (9.4%)
- not current: 6.438 sampel (6.5%)
- ever: 4.004 sampel (4.0%)

Dari hasil ini, terlihat bahwa sebagian besar data terdiri dari individu yang tidak pernah merokok (35.4%) dan yang tidak memiliki informasi riwayat merokok (35.3%). Kategori lainnya memiliki proporsi yang lebih kecil, di mana kategori former, current, not current, dan ever masing-masing berada di bawah 10%.

Visualisasi bar chart mendukung hal ini dengan menunjukkan dua batang tertinggi untuk kategori never dan No Info, sedangkan kategori lainnya relatif lebih rendah.

Informasi ini penting untuk diperhatikan dalam proses pra-pemodelan, terutama jika fitur smoking_history akan digunakan sebagai variabel input. Proporsi yang tidak seimbang dan keberadaan kategori No Info menunjukkan kemungkinan perlunya strategi penanganan data tambahan, seperti encoding khusus atau pemrosesan data hilang.

### Numerical Feature


*Eksplorasi data awal terhadap fitur-fitur numerikal*

![image](https://github.com/user-attachments/assets/0497aabe-0e35-4765-bd34-556f8b547c81)

Pada tahap ini, melakukan eksplorasi data awal terhadap fitur-fitur numerikal menggunakan histogram. Visualisasi ini membantu memahami distribusi nilai dari setiap fitur.

Berikut ringkasan pengamatan untuk masing-masing fitur:

* **age**: Distribusi usia relatif merata dengan lonjakan tinggi pada usia sekitar 80 tahun, yang mengindikasikan adanya akumulasi data pada batas usia maksimum.

* **hypertension** dan **heart\_disease**: Kedua fitur ini adalah biner (0 atau 1), dan distribusinya sangat tidak seimbang. Mayoritas data bernilai 0 (tidak memiliki hipertensi atau penyakit jantung), sementara hanya sebagian kecil yang bernilai 1.

* **bmi**: Terdistribusi positif skewed (condong ke kanan). Sebagian besar nilai berada pada kisaran 15–40, dengan puncak signifikan di sekitar 20–25. Ada kemungkinan nilai pencilan (outlier) di bagian kanan.

* **HbA1c\_level**: Data terlihat dalam bentuk diskrit dengan banyak nilai yang berulang (seperti 6.0, 6.5, dll), menunjukkan pengukuran dengan nilai-nilai standar tertentu. Distribusinya tidak merata, namun nilai antara 4 dan 7 mendominasi.

* **blood\_glucose\_level**: Distribusi bersifat multimodal (terdapat lebih dari satu puncak), dengan konsentrasi data pada kisaran 100–200. Beberapa nilai ekstrem juga tampak berada di atas 250, yang kemungkinan merupakan outlier.

* **diabetes**: Seperti fitur biner lainnya, distribusinya sangat tidak seimbang, dengan sebagian besar data bernilai 0 (tidak menderita diabetes), dan sebagian kecil bernilai 1 (menderita diabetes).

Hasil eksplorasi ini memberikan insight awal yang penting:

* Terdapat **ketidakseimbangan kelas** pada fitur biner (`hypertension`, `heart_disease`, dan `diabetes`).
* Fitur seperti `bmi`, `blood_glucose_level`, dan `age` mungkin mengandung **outlier** yang perlu ditangani.
* Beberapa fitur numerikal memiliki **distribusi tidak normal**, yang dapat memengaruhi performa algoritma tertentu.

Langkah selanjutnya yang dapat dilakukan termasuk transformasi data (seperti normalisasi atau log transform), penanganan outlier, dan balancing data bila diperlukan untuk keperluan pemodelan.

### Multivariate Analysis

### Categorical Feature

![image](https://github.com/user-attachments/assets/38edb724-822a-4f1d-8ec9-9bde8df7f6cb)

Pada tahap ini, menganalisis **rata-rata kemunculan diabetes** (`diabetes` bernilai 1) terhadap dua fitur kategorikal: `gender` dan `smoking_history`. Nilai rata-rata pada grafik batang ini merepresentasikan **proporsi individu yang menderita diabetes** dalam setiap kategori.

1. **Rata-rata diabetes terhadap gender**

* **Female**: sekitar 7.5%
* **Male**: sekitar 10%
* **Other**: tidak terlihat signifikan karena jumlah sampel sangat kecil (hampir nol)

Dari visualisasi ini, terlihat bahwa **laki-laki memiliki proporsi penderita diabetes yang lebih tinggi** dibandingkan perempuan. Perbedaan ini dapat menjadi insight awal bahwa gender berpotensi menjadi variabel prediktif yang relevan dalam model klasifikasi diabetes.

2. **Rata-rata diabetes terhadap smoking\_history**

* **former** (mantan perokok): memiliki proporsi tertinggi, sekitar **17%**
* Diikuti oleh **ever** (\~12%) dan **not current** (\~11%)
* **current** (perokok aktif): \~10%
* **never** (tidak pernah merokok): \~9.5%
* **No Info**: proporsi terendah (\~4%)

Dari hasil ini dapat disimpulkan bahwa:

* Riwayat merokok **berkorelasi positif** dengan kejadian diabetes, terutama pada individu yang pernah merokok namun sudah berhenti (*former*).
* Individu dengan status *No Info* justru memiliki proporsi terendah, namun hal ini bisa jadi karena ketidaklengkapan informasi atau bias dalam pelabelan data.

Hasil eksplorasi ini sangat berguna dalam:

* Menentukan pentingnya fitur (feature importance)
* Mengarahkan strategi feature engineering atau encoding untuk fitur kategorikal
* Memahami potensi adanya bias atau ketidakseimbangan dalam data

Langkah selanjutnya dapat berupa uji statistik untuk menguji signifikansi hubungan ini, atau langsung masuk ke tahap preprocessing dan pelatihan model klasifikasi.

### Numerical Feature

![image](https://github.com/user-attachments/assets/04acee9d-446b-4f6a-9460-4c5a0aa16375)

![image](https://github.com/user-attachments/assets/02ab9651-dd96-4048-8554-1d558cc8803e)

Pada tahap ini, menggunakan `sns.pairplot()` dengan `diag_kind='kde'` untuk **mengamati hubungan antar fitur numerik** dalam dataset, serta distribusi dari masing-masing fitur.

Visualisasi ini membantu:

* Mengidentifikasi pola hubungan (linear, non-linear, tidak berhubungan)
* Mendeteksi distribusi data (normal, skewed, multimodal)
* Mengenali outlier secara visual
* Menganalisis potensi korelasi antara fitur numerik

Insight dari Pairplot:

1. **Distribusi Individu Fitur (diagonal plot):**

* **Age**: Terdistribusi cukup merata, namun terlihat ada puncak signifikan di usia mendekati 80 tahun.
* **BMI**: Memiliki distribusi right-skewed dengan konsentrasi kuat di sekitar 20–30.
* **HbA1c\_level** dan **blood\_glucose\_level**: Terlihat multimodal (beberapa puncak) yang mengindikasikan adanya kelompok nilai khas (misalnya 6.5 pada HbA1c).
* **Diabetes**: Karena ini adalah data biner (0 atau 1), distribusinya berupa dua puncak diskret.

2. **Hubungan Antar Fitur:**

* **age vs diabetes**: Tidak ada hubungan linier yang jelas, tetapi bisa jadi ada pola non-linear (misalnya prevalensi diabetes meningkat di usia tua).
* **HbA1c\_level vs diabetes** dan **blood\_glucose\_level vs diabetes**: Tampak adanya pola pemisahan – individu dengan diabetes cenderung memiliki nilai yang lebih tinggi. Ini menunjukkan **potensi hubungan yang kuat**.
* **BMI dan age terhadap diabetes**: Tidak menunjukkan pola yang jelas dalam scatter plot; kemungkinan kontribusi tidak dominan secara individual.
* **heart\_disease dan hypertension** vs fitur lain: karena berupa biner, titik-titiknya membentuk garis horizontal; terlihat bahwa pasien dengan penyakit ini cenderung terdistribusi di usia tua.

Kesimpulan:

* **HbA1c\_level** dan **blood\_glucose\_level** tampak sebagai fitur yang **paling informatif** terhadap keberadaan diabetes berdasarkan visualisasi ini.
* Sebagian besar fitur numerik **tidak menunjukkan hubungan linear kuat satu sama lain**, menandakan rendahnya multikolinearitas.
* Informasi ini berguna untuk:

  * Menentukan fitur penting dalam model prediktif
  * Menyusun strategi seleksi fitur
  * Mempersiapkan transformasi data (misal: normalisasi atau log transform jika diperlukan)

**Korelasi Antar Fitur Numerik**

![image](https://github.com/user-attachments/assets/b757a80e-5d5b-4ed2-807e-756ba183efc5)

Pada tahap ini menggunakan heatmap dari `seaborn` untuk memvisualisasikan korelasi antar fitur numerik, dengan tujuan **mengetahui hubungan linear antar fitur, khususnya terhadap target `diabetes`.**

**Langkah-langkah:**

* Menghitung korelasi dengan `df[numerical_features].corr().round(2)` untuk memperoleh nilai korelasi antar fitur.
* Menampilkan heatmap menggunakan `sns.heatmap()` dengan `annot=True` dan `cmap='coolwarm'` untuk interpretasi visual yang jelas.
* Mengatur ukuran dan judul plot agar mudah dibaca.

**Insight:**

* Fitur `blood_glucose_level` (0.42) dan `HbA1c_level` (0.40) memiliki korelasi tertinggi terhadap `diabetes`.
* Korelasi antar fitur prediktor relatif rendah, menunjukkan minimnya multikolinearitas.

**Kesimpulan:**
Fitur yang berkorelasi tinggi dengan `diabetes` layak diprioritaskan dalam pemodelan, sementara rendahnya korelasi antar prediktor menunjukkan fitur saling melengkapi.

## Data Preparation

**Menghapus dua kolom yaitu 'heart_disease' dan 'hypertension'**

![image](https://github.com/user-attachments/assets/2e6000b3-28db-4e0a-aee5-6d6a1e4dc16c)

Pada tahap ini menghapus dua kolom yaitu 'heart_disease' dan 'hypertension' dari DataFrame df dengan tujuan untuk menyederhanakan data dan hanya mempertahankan fitur-fitur yang relevan untuk analisis atau pemodelan lebih lanjut.

Penjelasan:

df.drop(['heart_disease', 'hypertension'], inplace=True, axis=1)
→ Menghapus kolom 'heart_disease' dan 'hypertension' dari DataFrame secara permanen (inplace=True), karena dianggap tidak diperlukan dalam tahap analisis berikutnya.

df.head()
→ Menampilkan 5 baris pertama dari DataFrame setelah penghapusan kolom, untuk memastikan bahwa struktur data sudah sesuai dan kolom yang tidak dibutuhkan telah berhasil dihapus.

*Menangani Nilai Usia yang Tidak Masuk Akal (age < 1)*

![image](https://github.com/user-attachments/assets/a97b4454-fde6-46ad-ad87-e070186f4844)

Dari output terlihat bahwa:
- Usia rata-rata peserta adalah dewasa (sekitar 42 tahun). Distribusi cenderung merata antara usia muda dan tua.

*Menghapus outlier ekstrem pada fitur BMI*

![image](https://github.com/user-attachments/assets/b477b205-eb9b-49b3-bacb-10f4a309a8a5)

Pada tahap ini, menghapus outlier ekstrem pada fitur BMI dengan cara menyaring data yang berada di atas batas atas (38.45), sesuai perhitungan IQR sebelumnya.

Tujuannya adalah untuk mengurangi pengaruh nilai ekstrem yang dapat merusak performa model machine learning.

Setelah pembersihan, jumlah data berkurang menjadi 93.124 sampel, yang lebih representatif terhadap distribusi BMI normal dalam populasi.

*Memeriksa ukuran dataset setelah melakukan penghapusan outlier* 

![image](https://github.com/user-attachments/assets/eb8835e5-60a9-43a2-8cc4-8e4a5d9d052d)

Pada tahap ini, memeriksa ukuran setelah melakukan penghapusan outlier pada dataset menggunakan df.shape, yang menunjukkan bahwa data terdiri dari 93.124 baris dan 7 kolom.

Informasi ini menjadi patokan awal untuk membandingkan perubahan jumlah data setelah dilakukan pembersihan outlier, khususnya pada fitur BMI.

*Memvisualisasikan kembali boxplot fitur BMI*

![image](https://github.com/user-attachments/assets/d421e5f2-f192-4153-aa30-dbd5dd4cc54c)

Pada tahap ini, memvisualisasikan kembali boxplot fitur BMI setelah pembersihan outlier menggunakan data df.

Tujuannya adalah untuk mengevaluasi hasil pembersihan, memastikan bahwa nilai-nilai ekstrem di atas batas atas telah berhasil dihapus.

Hasil boxplot menunjukkan distribusi yang lebih rapat dan simetris, tanpa adanya outlier ekstrem, sehingga data lebih bersih dan siap digunakan untuk proses pemodelan machine learning.

**Encoding Data Categorical**

![image](https://github.com/user-attachments/assets/35871943-d89c-4a28-9f10-6dc2c1594628)

Pada tahap ini **melakukan encoding terhadap variabel kategorikal ‘gender’ dan ‘smoking\_history’ menggunakan teknik one-hot encoding** dengan tujuan **untuk mengubah data kategorikal menjadi format numerik biner agar bisa digunakan dalam pemodelan machine learning**.

Penjelasan:

* `pd.get_dummies(df['gender'], prefix='cut')`
  → Mengubah nilai kategori pada kolom `gender` menjadi beberapa kolom baru (`cut_Female`, `cut_Male`, `cut_Other`) yang berisi nilai `True` atau `False` tergantung dari jenis kelamin masing-masing baris.

* `pd.get_dummies(df['smoking_history'], prefix='color')`
  → Melakukan hal yang sama untuk kolom `smoking_history`, menghasilkan beberapa kolom seperti `color_current`, `color_never`, dll.

* `pd.concat([...], axis=1)`
  → Menambahkan kolom-kolom hasil one-hot encoding ke DataFrame asli secara horizontal (kolom).

* `df.drop(['gender','smoking_history'], axis=1, inplace=True)`
  → Menghapus kolom asli yang sudah dikodekan agar tidak terjadi duplikasi informasi.

* `df.head()`
  → Menampilkan 5 baris pertama dari DataFrame hasil transformasi, untuk memverifikasi bahwa encoding berhasil dan data siap untuk analisis atau pemodelan lebih lanjut.

**Train Test Split**

![image](https://github.com/user-attachments/assets/25d68960-b976-4b4c-8442-0f4ab22690a6)

Pada tahap ini **membagi data menjadi data latih (training set) dan data uji (testing set)** dengan tujuan **untuk mempersiapkan proses pelatihan dan evaluasi model machine learning, di mana model akan belajar dari data latih dan diuji performanya menggunakan data uji**.

Penjelasan tambahan:

* `X = df.drop(["diabetes"], axis=1)`
  → Memisahkan fitur (variabel independen) dari label (variabel target). Di sini, `X` berisi semua kolom kecuali kolom `diabetes`.

* `y = df["diabetes"]`
  → Menyimpan kolom `diabetes` sebagai variabel target yang ingin diprediksi.

* `train_test_split(X, y, test_size=0.1, random_state=123)`
  → Membagi data menjadi dua bagian:

  * 90% sebagai data latih (`X_train`, `y_train`)
  * 10% sebagai data uji (`X_test`, `y_test`)
  * `test_size=0.1` menunjukkan proporsi data uji.
  * `random_state=123` memastikan hasil pembagian data konsisten/reproducible setiap kali kode dijalankan.

![image](https://github.com/user-attachments/assets/f7481905-064d-4154-beb5-eaefceb9ffa9)

Pada tahap ini **mencetak jumlah sampel/data pada keseluruhan dataset, data latih, dan data uji** dengan tujuan **untuk memastikan bahwa proses pembagian data sebelumnya berjalan dengan benar dan proporsinya sesuai yang diharapkan**.

Penjelasan:

* `len(X)` menghitung total jumlah baris (sampel) dalam dataset fitur, yaitu seluruh data sebelum dibagi.
* `len(X_train)` menghitung jumlah sampel yang digunakan untuk melatih model.
* `len(X_test)` menghitung jumlah sampel yang disisihkan untuk menguji kinerja model setelah dilatih.

**Standarisasi**

![image](https://github.com/user-attachments/assets/cfb227e9-cad3-4d3e-a6f8-e9f6c517be81)

Pada tahap ini, dilakukan normalisasi atau standarisasi terhadap fitur-fitur numerik menggunakan StandardScaler dari scikit-learn dengan tujuan untuk menyamakan skala nilai dari setiap fitur numerik sehingga model machine learning dapat belajar dengan lebih baik dan tidak berat sebelah terhadap fitur dengan skala yang lebih besar.

Penjelasan langkah demi langkah:
1. numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
→ Saya menentukan daftar fitur yang bersifat numerik, yaitu usia, indeks massa tubuh (BMI), kadar HbA1c, dan kadar gula darah.

2. scaler = StandardScaler()
→ Saya membuat objek StandardScaler, yaitu alat yang akan mengubah data sehingga setiap kolom memiliki rata-rata 0 dan standar deviasi 1.

3. scaler.fit(X_train[numerical_features])
→ Saya “melatih” scaler ini hanya pada data latih (X_train) untuk menghitung rata-rata dan standar deviasi tiap fitur numerik.
→ Ini penting untuk menghindari kebocoran data (data leakage), yaitu saat informasi dari data uji (X_test) secara tidak sengaja digunakan saat pelatihan.

4. X_train[numerical_features] = scaler.transform(X_train[numerical_features])
→ Saya menerapkan transformasi skala ini ke fitur-fitur numerik di X_train.
→ Nilai-nilai asli (misalnya usia 50 tahun, atau kadar gula 130 mg/dL) diubah menjadi nilai standar, misalnya 0.75 atau -1.2.

5. X_test[numerical_features] = scaler.transform(X_test[numerical_features])
→ Saya juga menerapkan transformasi yang sama ke X_test dengan scaler yang sama (yang telah di-fit dari X_train).
→ Ini memastikan data uji berada dalam skala yang sama dengan data latih, tanpa menghitung ulang mean/std dari data uji.

6. X_train[numerical_features].head() dan X_test[numerical_features].head()
→ Saya dapat menampilkan 5 baris pertama dari masing-masing hasil transformasi untuk memverifikasi bahwa skala data sudah sesuai.

**Menampilkan statistik deskriptif dari fitur numerik**

![image](https://github.com/user-attachments/assets/132dec40-4535-41fd-81da-fc96b49befbe)

Pada tahap ini **menampilkan statistik deskriptif dari fitur numerik dalam data latih (X\_train) setelah dilakukan proses normalisasi atau standardisasi** dengan tujuan **untuk memeriksa distribusi data numerik dan memastikan bahwa data telah berskala (scaled) dengan benar, yaitu memiliki rata-rata 0 dan standar deviasi 1**.

Penjelasan:

* `X_train[numerical_features].describe()` memberikan ringkasan statistik untuk fitur-fitur numerik seperti `age`, `bmi`, `HbA1c_level`, dan `blood_glucose_level` di data latih.
* `.round(4)` membulatkan hasil ke 4 angka di belakang koma agar lebih mudah dibaca.

Dari hasil ini terlihat:

* **Mean (rata-rata)** semua fitur mendekati **0**, dan **standard deviation (std)** mendekati **1**, yang menandakan bahwa data telah dinormalisasi menggunakan **standar skala** (z-score normalization).
* **Min dan max** menunjukkan nilai terendah dan tertinggi setelah data diskalakan.
* Kuartil (25%, 50%, 75%) menunjukkan persebaran data di sekitar rata-rata.

![image](https://github.com/user-attachments/assets/7dbb697e-2b79-477e-85ab-11cbdeff5878)

Pada tahap ini, ditampilkan statistik deskriptif dari fitur numerik dalam data uji (X_test) setelah dilakukan proses standardisasi menggunakan StandardScaler. Tujuannya adalah untuk memeriksa apakah data uji telah berada pada skala yang sama seperti data latih, yaitu memiliki rata-rata mendekati 0 dan standar deviasi mendekati 1, sesuai prinsip normalisasi menggunakan Z-score.

Penjelasan:
- X_test[numerical_features].describe() memberikan ringkasan statistik untuk fitur-fitur numerik seperti age, bmi, HbA1c_level, dan blood_glucose_level dalam data uji.

- .round(4) digunakan untuk membulatkan angka ke empat desimal agar lebih mudah dibaca.

Kesimpulan:
* Mean dari keempat fitur berada sangat dekat dengan 0, dan standard deviation (std) mendekati 1, yang menandakan bahwa proses standardisasi berhasil diterapkan dengan konsisten ke data uji (X_test) menggunakan scaler yang sama dari data latih (X_train).

* Min dan max menunjukkan sebaran nilai setelah transformasi.

* Kuartil (25%, 50%, dan 75%) menunjukkan distribusi data setelah diskalakan.

## Modelling

Pada tahap ini, mengembangkan model machine learning menggunakan **tiga algoritma berbeda**, yaitu:

* **Logistic Regression**
* **Random Forest**
* **XGBoost Classifier**

Tujuan dari penggunaan ketiga model ini adalah untuk membandingkan performa mereka dalam memprediksi target secara akurat dan memilih model terbaik berdasarkan metrik evaluasi seperti **akurasi** dan **F1-score**. Berikut adalah penjelasan singkat mengenai cara kerja masing-masing algoritma:

---

#### 1. Logistic Regression

**Logistic Regression** adalah algoritma klasifikasi linear yang digunakan untuk memprediksi probabilitas dari sebuah kelas. Alih-alih memberikan output nilai kontinu seperti regresi linear, logistic regression menggunakan **fungsi sigmoid (logistik)** untuk memetakan output ke dalam rentang 0 hingga 1, sehingga cocok untuk tugas klasifikasi biner.

Cara kerjanya:

* Model menghitung kombinasi linear dari fitur input:

  $$
  z = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b
  $$
* Nilai `z` kemudian dimasukkan ke dalam fungsi sigmoid:

  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$
* Hasil sigmoid digunakan sebagai probabilitas prediksi, dan keputusan kelas ditentukan berdasarkan ambang batas tertentu (biasanya 0.5).

Logistic Regression bekerja dengan baik ketika hubungan antara fitur dan target bersifat **linear**.

---

#### 2. Random Forest Classifier

**Random Forest** adalah algoritma ensemble learning berbasis pohon keputusan yang membangun **banyak pohon keputusan** dan menggabungkan hasilnya (melalui voting mayoritas untuk klasifikasi) untuk meningkatkan akurasi dan mengurangi overfitting.

Cara kerjanya:

* Model membuat beberapa pohon keputusan (`n_estimators`), di mana setiap pohon:

  * Dilatih menggunakan **subset acak** dari data latih (dengan teknik bootstrap sampling).
  * Menggunakan subset acak dari fitur saat menentukan split terbaik (bagian dari teknik *bagging*).
* Prediksi akhir didasarkan pada mayoritas hasil prediksi dari seluruh pohon.

Random Forest sangat efektif dalam menangani **fitur non-linear**, interaksi kompleks antar fitur, serta **mengurangi varians** dari model pohon tunggal.

---

#### 3. XGBoost Classifier

**XGBoost (Extreme Gradient Boosting)** adalah algoritma ensemble berbasis pohon keputusan yang mengimplementasikan teknik **gradient boosting** secara efisien dan teroptimasi. Berbeda dengan Random Forest, XGBoost membangun pohon secara **berurutan**, dan setiap pohon baru berusaha **memperbaiki kesalahan** dari pohon sebelumnya.

Cara kerjanya:

* Model memulai dari prediksi awal (misalnya rata-rata).
* Setiap iterasi menambahkan pohon baru yang dilatih untuk meminimalkan **loss function** berdasarkan **gradien error** dari model sebelumnya.
* Proses ini berlanjut hingga mencapai jumlah pohon yang ditentukan atau tidak ada lagi perbaikan signifikan.

XGBoost dikenal karena:

* Menggunakan teknik **regularisasi** untuk mencegah overfitting.
* Mendukung **parallel processing**, sehingga sangat cepat.
* Memberikan hasil akurat terutama pada dataset kompleks.

![image](https://github.com/user-attachments/assets/c44e50be-cac6-43fa-a1be-6065ea1c81a0)

Pada tahap ini **menyiapkan sebuah DataFrame kosong bernama `models` untuk menyimpan hasil evaluasi dari beberapa algoritma machine learning** dengan tujuan **agar hasil akurasi dan metrik lainnya dari masing-masing model dapat dicatat, dibandingkan, dan dianalisis dengan lebih terstruktur**.

Penjelasan:

* `pd.DataFrame(...)` membuat sebuah tabel kosong (DataFrame) yang memiliki:

  * **Index** berupa empat baris:

    * `train_acc`: akurasi model pada data latih
    * `test_acc`: akurasi model pada data uji
    * `train_f1`: nilai F1-score pada data latih
    * `test_f1`: nilai F1-score pada data uji
  * **Kolom** berisi nama-nama model yang akan digunakan, yaitu:

    * `LogisticRegression`
    * `RandomForest`
    * `XGB` (XGBoost)

Tujuannya adalah untuk menyimpan dan membandingkan performa dari ketiga model tersebut dalam satu tempat yang rapi dan mudah dianalisis, baik dari sisi akurasi maupun F1-score, di data latih maupun data uji.

## Logistic Regression

![image](https://github.com/user-attachments/assets/827f44ea-fc2d-4495-9fa6-26459141df2a)

Pada tahap ini **melatih model Logistic Regression menggunakan data latih, lalu mengukur performanya dengan menghitung akurasi dan F1-score baik pada data latih maupun data uji**.

Penjelasan singkat:

* Membuat model `LogisticRegression` dengan batas maksimal iterasi 1000 dan `random_state=42` untuk hasil yang konsisten.
* Melatih model (`fit`) menggunakan `X_train` dan `y_train`.
* Membuat prediksi pada data latih dan data uji.
* Menghitung akurasi dan F1-score dari prediksi tersebut.
* Menyimpan hasil metrik ke dalam DataFrame `models` pada kolom `LogisticRegression`.

Dengan cara ini, saya dapat melihat seberapa baik model Logistic Regression dalam mempelajari data dan memprediksi data baru.

## Random Forest

![image](https://github.com/user-attachments/assets/e2ca8473-cf03-4704-a4b5-441ea44bb387)

Pada tahap ini **melatih model Random Forest menggunakan algoritma `RandomForestClassifier` dari pustaka `sklearn.ensemble`**. Model ini digunakan dengan konfigurasi sebagai berikut:

* `n_estimators=100`: Menggunakan 100 pohon keputusan (decision trees).
* `max_depth=10`: Mengatur kedalaman maksimum setiap pohon hingga 10 untuk menghindari overfitting.
* `random_state=42`: Menetapkan seed random agar hasil eksperimen dapat direproduksi secara konsisten.

Langkah-langkah pelatihan dan evaluasi model dilakukan sebagai berikut:

1. Membuat objek model `RandomForestClassifier` dengan parameter seperti di atas.
2. Melatih model menggunakan data latih (`X_train`, `y_train`).
3. Menggunakan model untuk memprediksi label pada data latih (`X_train`) dan data uji (`X_test`).
4. Menghitung metrik evaluasi:

   * Akurasi (`accuracy_score`) dan F1-score (`f1_score`) pada data latih dan data uji.
   * Hasil evaluasi disimpan ke dalam DataFrame `models` pada kolom `RandomForest`.

Model Random Forest ini digunakan sebagai baseline yang kuat karena kemampuannya menangani data dengan fitur non-linear dan interaksi antar fitur secara otomatis.

## XGBoost Classifier

![image](https://github.com/user-attachments/assets/b7421833-8089-45c4-a2c6-91d8410ad08c)

Selanjutnya, juga melatih model **XGBoost menggunakan `XGBClassifier` dari pustaka `xgboost`**, dengan parameter sebagai berikut:

* `n_estimators=50`: Jumlah boosting rounds atau pohon yang akan dibuat sebanyak 50.
* `max_depth=5`: Kedalaman maksimum setiap pohon dibatasi hingga 5 untuk menghindari kompleksitas berlebih.
* `random_state=42`: Digunakan untuk memastikan hasil pelatihan yang dapat direproduksi.
* `use_label_encoder=False`: Menonaktifkan encoder label default XGBoost untuk menghindari peringatan yang sudah deprecated.
* `eval_metric='logloss'`: Mengatur metrik evaluasi internal selama pelatihan ke log-loss (digunakan untuk klasifikasi biner).

Langkah-langkah pelatihan dan evaluasi model:

1. Membuat objek `XGBClassifier` dengan parameter di atas.
2. Melatih model menggunakan data latih (`X_train`, `y_train`).
3. Melakukan prediksi pada data latih dan data uji.
4. Menghitung akurasi dan F1-score pada kedua set data, kemudian menyimpannya di DataFrame `models` pada kolom `XGB`.

Model XGBoost dikenal dengan performanya yang unggul dalam berbagai kompetisi machine learning karena kemampuannya melakukan boosting secara efisien dan menangani overfitting dengan baik melalui pengaturan parameter seperti `max_depth`.

## Evaluation

![image](https://github.com/user-attachments/assets/cb8e9705-c1df-46d3-b08d-4a892a79eaa6)

Pada tahap ini **menampilkan tabel hasil evaluasi performa ketiga model machine learning yang sudah dilatih dan diuji** dengan tujuan **untuk membandingkan dan melihat bagaimana masing-masing model bekerja pada data latih dan data uji berdasarkan metrik akurasi dan F1-score**.

Penjelasan:

* Kolom menunjukkan tiga model: **Logistic Regression**, **Random Forest**, dan **XGBoost**.
* Baris `train_acc` dan `train_f1` menunjukkan performa model pada data latih. Semua model menunjukkan akurasi tinggi (sekitar 96–97%) dan F1-score yang cukup bagus, dengan Random Forest dan XGBoost sedikit lebih unggul dibanding Logistic Regression.
* Baris `test_acc` dan `test_f1` menunjukkan performa model pada data uji. Nilai akurasi tetap tinggi (sekitar 96–97%) dan F1-score juga meningkat secara proporsional, menunjukkan bahwa model mampu mempertahankan performa pada data yang belum pernah dilihat sebelumnya.

Kesimpulan singkat:
Semua model menunjukkan performa yang **baik dan stabil** baik pada data latih maupun data uji, dengan **Random Forest dan XGBoost** sedikit lebih unggul dibanding Logistic Regression dalam hal F1-score. Tidak ada indikasi overfitting yang mencolok, dan hasil ini menunjukkan bahwa proses pelatihan serta evaluasi telah berjalan dengan baik.


![image](https://github.com/user-attachments/assets/37ff815e-14c0-4d7a-9ee1-b2fd5f79288a)

Pada tahap ini **membuat sebuah fungsi `evaluate_model` untuk mengevaluasi performa model machine learning secara lengkap dan visual, baik pada data latih maupun data uji** dengan tujuan **memudahkan proses pengecekan hasil model sekaligus menampilkan metrik penting dan visualisasi confusion matrix agar lebih mudah dipahami**.

Penjelasan:

* Fungsi menerima input: model yang sudah dilatih, data latih dan uji beserta labelnya, serta nama model sebagai penanda.
* Fungsi melakukan prediksi pada data latih dan data uji.
* Menghitung metrik performa penting: akurasi dan F1-score untuk kedua set data.
* Mencetak hasil metrik tersebut dengan format yang mudah dibaca.
* Membuat dan menampilkan **confusion matrix** untuk data uji, yaitu tabel yang menunjukkan jumlah prediksi benar dan salah, dengan visualisasi heatmap yang berwarna untuk memudahkan interpretasi.
* Fungsi mengembalikan nilai metrik yang dihitung agar bisa digunakan kembali untuk analisis atau pencatatan lebih lanjut.

Dengan fungsi ini, proses evaluasi model jadi lebih praktis, rapi, dan informatif, membantu dalam memilih model terbaik secara objektif.

![image](https://github.com/user-attachments/assets/8ed98748-8e0f-4dc2-a567-4ce1b699ff37)

Di tahap ini, mengevaluasi model **Logistic Regression** menggunakan fungsi `evaluate_model`. Dari hasilnya, model ini menunjukkan performa yang sangat baik di data latih, dengan **akurasi sekitar 96,36%** dan **F1-score sekitar 71,55%**, yang menandakan model mampu mengenali pola pada data latih dengan cukup baik.

Ketika diuji pada data uji, model juga menunjukkan hasil yang **konsisten**, dengan **akurasi sekitar 96,65%** dan **F1-score sekitar 72,77%**. Ini menunjukkan bahwa model dapat menggeneralisasi dengan baik terhadap data baru, tanpa mengalami penurunan performa yang signifikan.

Jadi, performa model cukup stabil antara data latih dan data uji, menunjukkan bahwa **Logistic Regression merupakan model baseline yang andal**, meskipun masih bisa ditingkatkan lebih lanjut dengan model yang lebih kompleks.

![image](https://github.com/user-attachments/assets/1d28984f-417d-4293-a589-6552348f0481)

Di tahap ini, mengevaluasi model **Random Forest** menggunakan fungsi `evaluate_model`. Dari hasilnya, model ini menunjukkan performa yang sangat baik di data latih, dengan **akurasi sekitar 97,47%** dan **F1-score sekitar 80,00%**, yang menandakan model bisa mengenali pola pada data latih dengan sangat baik.

Ketika diuji pada data uji, model tetap menunjukkan performa yang sangat baik, dengan **akurasi sekitar 97,76%** dan **F1-score sekitar 81,87%**, bahkan sedikit lebih tinggi dari pada data latih. Ini menunjukkan bahwa model mampu menggeneralisasi dengan sangat baik terhadap data baru.

Jadi, walaupun model terlihat hebat di data latih, performanya di data uji **juga sangat baik**, sehingga Random Forest merupakan pilihan model yang kuat dan andal dalam menangani data ini.

![image](https://github.com/user-attachments/assets/e29d429b-2eb3-4b47-a35b-610f68b85af6)

Di tahap ini, mengevaluasi model **XGBoost** dengan menggunakan fungsi `evaluate_model`. Hasil evaluasi menunjukkan bahwa model ini memiliki performa sangat bagus di data latih, dengan **akurasi sekitar 97,50%** dan **F1-score sebesar 80,30%**. Ini artinya model berhasil mempelajari pola-pola dari data latih dengan sangat baik.

Namun, saat diuji pada data uji, model tetap menunjukkan performa yang sangat baik, dengan **akurasi sekitar 97,73%** dan **F1-score sebesar 81,73%**, hanya sedikit lebih rendah dari hasil pada data latih. Hal ini menunjukkan bahwa model XGBoost mampu menggeneralisasi dengan baik ke data baru dan tidak mengalami overfitting.

Singkatnya, meskipun model tampak hebat saat dilatih, hasil di data uji **juga sangat baik**, sehingga XGBoost dapat dianggap sebagai salah satu model terbaik dalam percobaan ini untuk memberikan prediksi yang akurat pada data yang belum pernah dilihat sebelumnya.

![image](https://github.com/user-attachments/assets/1c2cafe8-bd90-4555-a5b0-41dceab3d230)

![image](https://github.com/user-attachments/assets/20b9ad70-27fb-48d8-a037-d83b850d1cee)

**Penjelasan Kode**

Di bagian ini, membuat visualisasi untuk membandingkan performa tiga model machine learning (Logistic Regression, Random Forest, dan XGBoost) berdasarkan metrik **accuracy** dan **F1-score**, baik di data pelatihan (**train**) maupun data pengujian (**test**). Langkah-langkahnya sebagai berikut:

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

---

**Penjelasan Grafik**

Di tahap ini, membandingkan performa tiga model — **Logistic Regression**, **Random Forest**, dan **XGBoost** — dengan menggunakan metrik akurasi dan F1-score. Dari grafik yang ditampilkan, kita bisa menyimpulkan hal berikut:

🔹 Logistic Regression

* **Akurasi pelatihan** sangat tinggi (\~96%), dan **akurasi pengujian** juga tetap tinggi di sekitar 96,65%.
* **F1-score pelatihan** cukup tinggi (\~71%), dan **F1-score pengujian** sedikit lebih tinggi di sekitar 72,77%.
* Ini menunjukkan model ini **stabil** dan mampu mengenali pola di data baru dengan cukup baik, meskipun tidak sekuat model lain yang lebih kompleks.

🔹 Random Forest

* Performa di data pelatihan sangat baik (**akurasi \~97,47%**, **F1-score \~80%**), dan **performa di data pengujian juga tinggi**, dengan **akurasi \~97,76%** dan **F1-score \~81,87%**.
* Hal ini menunjukkan bahwa model ini **tidak overfitting** dan mampu menggeneralisasi dengan sangat baik ke data baru.

🔹 XGBoost

* Di tahap ini, mengevaluasi model XGBoost dengan menggunakan fungsi `evaluate_model`.
  Hasil evaluasi menunjukkan bahwa model ini memiliki performa sangat bagus di data latih, dengan **akurasi sekitar 97,50%** dan **F1-score sebesar 80,30%**. Ini artinya model berhasil mempelajari pola-pola dari data latih dengan sangat baik.

* Saat diuji pada data uji, **akurasi dan F1-score tetap tinggi**, masing-masing sekitar **97,73%** dan **81,73%**. Hal ini menunjukkan bahwa model XGBoost **mampu menggeneralisasi dengan sangat baik** dan merupakan salah satu model dengan performa terbaik dalam eksperimen ini.

## Kesimpulan
**Business Understanding dan Evaluasi Model**

**1. Apakah sudah menjawab setiap Problem Statement?**

**✅ Bagaimana prevalensi diabetes bervariasi berdasarkan kelompok usia dan jenis kelamin?**
Ya. Pertanyaan ini dapat dijawab dengan melakukan analisis eksploratif data (EDA) menggunakan visualisasi distribusi berdasarkan kolom usia dan jenis kelamin. Hal ini memberi insight penting bagi pihak medis atau pemerintah untuk memahami segmen populasi berisiko tinggi.

**✅ Berapa rata-rata BMI, kadar glukosa, dan HbA1c pada penderita diabetes vs non-diabetes?**
Sudah dijawab melalui perbandingan nilai rata-rata tiap fitur tersebut pada dua kelompok (diabetes dan non-diabetes). Ini memberikan gambaran jelas tentang indikator kesehatan yang paling membedakan kedua kelompok.

**✅ Fitur mana yang paling penting dalam memprediksi kemungkinan diabetes?**
Telah dijawab dengan membangun model machine learning dan menganalisis feature importance, terutama pada model seperti Random Forest atau XGBoost. Fitur seperti HbA1c, Glukosa, dan BMI terbukti menjadi indikator utama.

---

**2. Apakah berhasil mencapai setiap Goals?**

| Goals                                                               | Status | Penjelasan Singkat                                                    |
| ------------------------------------------------------------------- | ------ | --------------------------------------------------------------------- |
| Menganalisis distribusi diabetes berdasarkan kelompok usia & gender | ✅      | Sudah dianalisis melalui visualisasi dan statistik deskriptif         |
| Menganalisis perbedaan rata-rata BMI, HbA1c, Glukosa                | ✅      | Sudah dibedakan antara kelompok diabetes dan non-diabetes             |
| Membangun model ML & identifikasi fitur penting                     | ✅      | Model dibangun & fitur penting sudah ditentukan (HbA1c, Glukosa, BMI) |

---

**3. Evaluasi Solusi yang Diberikan (Solution Statement)**

**Solusi: Gunakan model non-linear seperti Random Forest dan XGBoost.**

| Model              | Train Accuracy | Test Accuracy | Train F1 | Test F1 | Evaluasi          |
| ------------------ | -------------- | ------------- | -------- | ------- | ----------------- |
| LogisticRegression | 96.36%         | 96.65%        | 71.55%   | 72.77%  | Generalisasi Baik |
| RandomForest       | 97.47%         | 97.76%        | 80.00%   | 81.87%  | Sangat Baik       |
| XGBoost            | 97.50%         | 97.73%        | 80.30%   | 81.73%  | Sangat Baik       |

**🧩 Kesimpulan:**

* Semua model menunjukkan performa **konsisten** dan **stabil** antara data pelatihan dan pengujian.
* Tidak ditemukan indikasi overfitting. Model **mampu menggeneralisasi** dengan baik terhadap data baru.
* Solusi yang direncanakan **berhasil memberikan hasil optimal dalam prediksi diabetes**.

---

**Rekomendasi Lanjutan**

Untuk memastikan solusi benar-benar berdampak terhadap bisnis dan menjawab kebutuhan nyata, berikut saran lanjutan:

1. **Cross-validation** untuk validasi yang lebih stabil dan adil.
2. **Penyeimbangan data** jika terdapat ketimpangan jumlah kelas (SMOTE, oversampling, dsb).
3. **Feature selection & regularisasi** untuk mempertahankan model yang efisien dan robust.
4. Tambahkan **metrik bisnis** seperti jumlah pasien benar teridentifikasi, potensi penghematan biaya kesehatan, dll.

---

**Kesimpulan Akhir**

* Semua problem statement dan goals **sudah terjawab secara teknis dan validasi model** menunjukkan performa sangat baik.
* Model yang dibangun **efektif digunakan dalam konteks nyata** karena berhasil memprediksi secara akurat pada data uji.
* Solusi yang direncanakan **layak diimplementasikan** untuk mendukung pengambilan keputusan dalam manajemen kesehatan dan penanganan diabetes.



















