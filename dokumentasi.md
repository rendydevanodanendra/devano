# Deteksi Outlier Menggunakan KNN dengan Euclidean Distance

**Penulis**: AI Assistant  
**Tanggal**: 2023-10-05  

---

## Daftar Isi
1. [Konsep Dasar](#konsep-dasar)  
2. [Langkah Implementasi](#langkah-implementasi)  
3. [Contoh Kode Python](#contoh-kode-python)  
4. [Interpretasi Hasil](#interpretasi-hasil)  
5. [Keuntungan & Keterbatasan](#keuntungan--keterbatasan)  
6. [Referensi](#referensi)  

---

## Konsep Dasar

### 1. KNN untuk Outlier Detection
K-Nearest Neighbors (KNN) adalah algoritma berbasis jarak yang mengidentifikasi outlier berdasarkan kedekatan suatu titik data dengan tetangga terdekatnya.  
- **Outlier**: Titik data yang jaraknya jauh dari sebagian besar data lain.  
- **Euclidean Distance**: Metrik jarak antara dua titik dalam ruang multidimensi.  

### 2. Intuisi Matematis
- **Rumus Euclidean Distance** untuk titik \( P(x_1, x_2, ..., x_n) \) dan \( Q(y_1, y_2, ..., y_n) \):  
  \[
  d(P, Q) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
  \]
- **Threshold**: Batas statistik (misalnya, persentil ke-95) untuk menentukan outlier.  

---

## Langkah Implementasi

1. **Pilih Nilai \( k \)**  
   Tentukan jumlah tetangga terdekat (misal: \( k = 3 \) atau \( k = 5 \)).  

2. **Hitung Jarak Euclidean**  
   Untuk setiap titik data, hitung jarak ke \( k \)-tetangga terdekat.  

3. **Hitung Rata-Rata Jarak**  
   Rata-rata jarak \( k \)-tetangga terdekat menjadi indikator "keterasingan" suatu titik.  

4. **Tentukan Threshold**  
   Gunakan persentil (misal: 95%) atau standar deviasi untuk menetapkan batas outlier.  

5. **Identifikasi Outlier**  
   Titik dengan rata-rata jarak > threshold dianggap outlier.  

---

## Contoh Kode Python

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Dataset contoh
X = np.array([[1, 2], [2, 3], [3, 4], [10, 12], [11, 13], [12, 14]])

# Inisialisasi KNN dengan k=3
k = 3
knn = NearestNeighbors(n_neighbors=k)
knn.fit(X)

# Hitung jarak ke k-tetangga terdekat
distances, indices = knn.kneighbors(X)

# Hitung rata-rata jarak per titik
avg_distances = distances.mean(axis=1)

# Tentukan threshold (persentil ke-95)
threshold = np.percentile(avg_distances, 95)

# Identifikasi outlier
outliers = X[avg_distances > threshold]

# Visualisasi
plt.scatter(X[:, 0], X[:, 1], label='Data Normal')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outlier')
plt.legend()
plt.title('Deteksi Outlier dengan KNN')
plt.show()