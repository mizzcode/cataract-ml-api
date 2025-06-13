# Machine Learning API

Machine Learning API ini dibangun menggunakan FastAPI untuk memproses gambar dan mengembalikan hasil deteksi menggunakan model machine learning yang telah dilatih sebelumnya. API ini dapat menerima upload gambar, melakukan prediksi, dan mengembalikan hasil dalam format JSON terstruktur.

## Project Structure

```
.
├── .gitattributes
├── .gitignore
├── Dockerfile
├── ML_CataractSense.ipynb
├── README.md
├── requirements.txt
├── run.py
├── model/
│   └── model_vgg16.keras
└── app/
    ├── main.py
    ├── models/
    │   └── model_loader.py
    ├── routes/
    │   └── predict.py
    ├── schemas/
    │   └── prediction.py
    └── utils/
        ├── image_processing.py
        └── numpy_compat.py
```

- **app/main.py**: Entry point aplikasi FastAPI.
- **app/models/model_loader.py**: Modul untuk load dan inisialisasi model ML.
- **app/routes/predict.py**: Endpoint API untuk prediksi gambar.
- **app/utils/image_processing.py**: Utilitas untuk pemrosesan gambar.
- **app/schemas/prediction.py**: Skema data untuk request dan response prediksi.
- **model/model_vgg16.keras**: File model machine learning yang telah dilatih.
- **requirements.txt**: Daftar dependensi Python.
- **run.py**: Alternatif entry point untuk menjalankan aplikasi.

## Setup Instructions

1. **Clone repository:**
   ```sh
   git clone <repository-url>
   cd Machine_Learning_Api
   ```

2. **Buat virtual environment (opsional tapi direkomendasikan):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi:**
   ```sh
   uvicorn app.main:app --reload
   ```
   Atau, jika menggunakan `run.py`:
   ```sh
   python run.py
   ```

## Usage

### Melakukan Prediksi

Kirim permintaan POST ke endpoint `/predict` dengan file gambar pada form-data.

**Contoh menggunakan `curl`:**
```sh
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_your_image.jpg"
```

### Response

API akan mengembalikan response JSON berisi hasil prediksi, misalnya:
```json
{
  "prediction": "Cataract",
  "confidence": 0.98
}
```

## Docker (Opsional)

Jika ingin menjalankan aplikasi menggunakan Docker:

1. **Build image:**
   ```sh
   docker build -t ml-api .
   ```
2. **Run container:**
   ```sh
   docker run -p 8000:8000 ml-api
   ```

## Notebook

- **ML_CataractSense.ipynb**: Notebook untuk eksplorasi dan pengujian model secara interaktif.

## License

Proyek ini dilisensikan di bawah MIT License. Lihat file LICENSE untuk detail lebih lanjut.