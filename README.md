# Laporan Proyek Machine Learning - Dea Yuliani Sabrina

## Domain Proyek

### Latar Belakang
Diabetes mellitus adalah salah satu penyakit tidak menular yang paling banyak menyebabkan kematian secara global. Menurut data dari World Health Organization (WHO), sekitar 422 juta orang di seluruh dunia hidup dengan diabetes, dan angka ini terus meningkat setiap tahunnya. Di Indonesia sendiri, prevalensi diabetes terus naik signifikan, mencapai 11% pada tahun 2021 menurut Riset Kesehatan Dasar (Riskesdas) Kementerian Kesehatan.

Deteksi dini terhadap risiko diabetes menjadi sangat penting agar pasien dapat segera melakukan perubahan gaya hidup atau intervensi medis untuk mencegah komplikasi lebih lanjut seperti penyakit jantung, gagal ginjal, atau kebutaan.

Namun, diagnosis dini diabetes sering kali sulit dilakukan tanpa tes laboratorium khusus seperti pemeriksaan HbA1c atau kadar glukosa darah, yang tidak selalu tersedia terutama di daerah terpencil. Di sinilah peran teknologi, terutama machine learning, menjadi relevan dan bermanfaat. Dengan memanfaatkan data kesehatan dasar pasien, kita dapat membangun model prediksi untuk mengidentifikasi individu yang berisiko terkena diabetes.

Referensi : [World Health Organization. "Diabetes." WHO, 2023](https://www.who.int/news-room/fact-sheets/detail/diabetes)

## Business Understanding

### Problem Statements

Proyek ini dibangun untuk lembaga di bidang kesehatan preventif dengan karakteristik bisnis sebagai berikut:
+ Lembaga berencana mengembangkan model prediksi risiko diabetes berbasis data kesehatan dasar agar bisa digunakan masyarakat umum dan tenaga medis sebagai alat bantu diagnosis awal.
+ Lembaga ingin meningkatkan efektivitas deteksi risiko diabetes di masyarakat luas, khususnya di wilayah dengan keterbatasan akses fasilitas medis, dengan pendekatan berbasis machine learning.

### Problem Statement
1. Apakah seseorang berisiko terkena diabetes berdasarkan fitur kesehatannya?
2. Fitur apa saja yang paling memengaruhi risiko terkena diabetes?
3. Bagaimana cara memproses data agar model dapat belajar secara optimal?

### Goals
1. 
2.
3. 

### Solution Statement
1. 

## Data Understanding 

Dataset yang digunakan dalam proyek ini merupakan data harga sewa rumah dengan berbagai karakteristik di India. Dataset ini dapat diunduh di [Kaggle : Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 100000 sample dengan 9 fitur.
+ Dataset memiliki 4 fitur bertipe int64, 2 fitur bertipe object, dan 3 fitur bertipe float64.
+ Tidak ada missing value dalam dataset.

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

