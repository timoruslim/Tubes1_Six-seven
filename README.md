<div align="center">

# Feed-Forward Neural Network

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2-4DABCF?logo=numpy&logoColor=fff)
![SciPy](https://img.shields.io/badge/SciPy-1.17.1-8CA1E5?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557c?logo=python&logoColor=white)

</div>

## 📘 Deskripsi

Di sini, kami membuat package untuk mengimplementasikan **Multi-Layer Perceptron** (MLP) atau **Feed-Forward Neural Network** (FFNN). Package secara efisien dan modular menerapkan _forward pass_, _backpropagation_, dan optimisasi bobot yang menjadi pokok mekanisme MLP. Dengan pacakage ini, pengguna dapat membangun MLP sendiri dengan sangat mudah, seperti `Scikit`, tetapi dengan arsitektur yang sangat _customizble_, seperti `Pytorch` atau `Tensor Flow`.

---

## ✨ Fitur Utama

- 📐 **Automatic Differentiation**
   - Kelas `Tensor` yang bisa menyimpan data dan gradien.
   - Pembentukan graf komputasi berisi `Tensor` untuk memudahkan _forward pass_ dan _backpropagation_.
- 🧩 **Customizble Layer**
   - Spesifikasi jumlah neuron, jenis aktivasi, inisialisasi bobot, dan regularisasi.
   - Implementasi RMSNorm.
- 🎯 **Efficient Fitting**
   - Spesifikasi _learning rate_, fungsi loss, dan metode optimisasi.
   - Implementasi Adam Optimizer.
- 🔍 **Model Transparency**
   - Pencatatan rinci nilai dan gradien bobot tiap layer.
   - Memungkinkan visualisasi, penyimpanan, dan pemuatan bobot.

---

## 📁 Struktur Folder

```

Tubes1_Six-Seven/
├── data/
│   └── datasetml_2026.csv
├── doc/
│   └── Tubes1_K1_Six-seven.pdf
├── src/
│   ├── ffnn/
│   │   ├── activation.py
│   │   ├── engine.py
│   │   ├── initialize.py
│   │   ├── loss.py
│   │   ├── model.py
│   │   ├── nn.py
│   │   └── optimizer.py
│   └── pengujian.ipynb
└── README.md

```

---

## ⚙️ Requirement & Instalasi

### Prasyarat

- python ≥ 3.13
- numpy ≥ 2.2
- scipy ≥ 1.17.1
- matplotlib ≥ 3.10
- tqdm ≥ 4.67.3
- ipywidgets ≥ 8.1.8
- ipykernel (untuk notebook)

### Instalasi

1. Clone repository.

   ```bash
   git clone https://github.com/timoruslim/Tubes1_Six-Seven.git
   cd Tubes1_Six-Seven
   ```

2. Pasang dependensi.

   ```
   pip install -r requirements.txt
   ```

   Jika file `requirements.txt` belum ada, bisa instal manual.

   ```
   pip install numpy scipy matplotlib tqdm ipywidgets jupyter
   ```

---

## 🚀 Menggunakan Package

### 1. Pertama

Lakukan ini.

### 2. Kedua

Lakukan itu.

---

## 👨‍💻 Author

| Nama                 | NIM      |
| -------------------- | -------- |
| Albi Arrizkya Putra  | 10122062 |
| Timothy Niels Ruslim | 10123053 |

---

## 🔗 Tautan

- 📂 [Repository GitHub](https://github.com/timoruslim/Tubes1_Six-seven)

---

> Dibuat sebagai bagian dari Tugas Besar 1 IF3270 Pembelajaran Mesin 2026 – ITB
