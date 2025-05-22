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
1. Faktor-faktor (fitur) apa yang paling memengaruhi prediksi risiko diabetes seseorang?
2. Bagaimana cara memprediksi apakah seseorang berisiko terkena diabetes berdasarkan data kesehatan dasar tanpa tes laboratorium?
3. Model klasifikasi mana yang paling akurat untuk memprediksi risiko diabetes pada dataset ini?

### Goals
1. Membangun model prediksi risiko diabetes berdasarkan data kesehatan dasar (tanpa memerlukan hasil laboratorium), sehingga dapat digunakan untuk skrining dini dan membantu pengambilan keputusan medis awal.
2. Mengevaluasi dan membandingkan performa beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan XGBoost untuk mengetahui model mana yang memberikan hasil terbaik dalam mendeteksi risiko diabetes.
3. Mengidentifikasi fitur-fitur paling signifikan dalam memengaruhi risiko diabetes.

### Solution Statement
1. Membangun model klasifikasi baseline menggunakan Logistic Regression dan Random Forest untuk memprediksi apakah seseorang berisiko terkena diabetes atau tidak.

