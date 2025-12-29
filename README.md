# 🌱 Productivity Predictor API - Dokumentasi

API untuk prediksi produktivitas pertanian menggunakan model XGBoost yang sudah di-training.

## 📋 Daftar Isi

- [Setup & Installation](#setup--installation)
- [Menjalankan API](#menjalankan-api)
- [Endpoint API](#endpoint-api)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Alur Preprocessing](#alur-preprocessing)

---

## 🚀 Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Model

Jika Anda sudah punya model dari Colab, letakkan file `model.pkl` di folder `saved/`.

Atau, train model baru dengan data lokal:

```bash
python convert_model.py
```

Script ini akan:

- Membaca data training dari CSV
- Training model XGBoost dengan preprocessing yang sama seperti di Colab
- Menyimpan model ke `saved/model.pkl`

---

## ▶️ Menjalankan API

```bash
python main.py
```

atau

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API akan berjalan di: `http://localhost:8000`

Dokumentasi interaktif (Swagger UI): `http://localhost:8000/docs`

---

## 📚 Endpoint API

### 1. **GET /** - Root Endpoint

Informasi dasar tentang API

**Response:**

```json
{
  "message": "Productivity Predictor API",
  "status": "Model loaded ✅",
  "endpoints": {
    "predict": "/predict/ - Prediksi tunggal",
    "batch_predict": "/batch_predict/ - Prediksi batch",
    "health": "/health/ - Cek status API",
    "reload_model": "/reload_model/ - Reload model",
    "model_info": "/model_info/ - Info tentang model"
  }
}
```

### 2. **GET /health/** - Health Check

Cek status kesehatan API dan model

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "XGBoost Regressor",
  "preprocessing": {
    "scaler": "StandardScaler",
    "encoder": "OneHotEncoder"
  }
}
```

### 3. **GET /model_info/** - Model Information

Informasi detail tentang model dan feature importance

**Response:**

```json
{
  "model_type": "XGBRegressor",
  "is_fitted": true,
  "n_features": 5,
  "feature_importance": {
    "elevasi_mdpl": 0.25,
    "suhu_c": 0.3,
    "curah_hujan_mm_per_day": 0.35,
    "pola_monokultur": 0.05,
    "pola_polikultur": 0.05
  },
  "pola_tanam_categories": ["monokultur", "polikultur"],
  "preprocessing_pipeline": {
    "step_1": "OneHotEncoding untuk pola_tanam",
    "step_2": "Combine numeric features dengan encoded pola_tanam",
    "step_3": "StandardScaler untuk normalisasi"
  }
}
```

### 4. **POST /predict/** - Single Prediction

Prediksi produktivitas untuk satu set data

**Request Body:**

```json
{
  "elevasi_mdpl": 836,
  "suhu_c": 22.4,
  "curah_hujan_mm_per_day": 110,
  "pola_tanam": "polikultur"
}
```

**Response:**

```json
{
  "produktivitas_pred": 1850.25,
  "input_data": {
    "elevasi_mdpl": 836,
    "suhu_c": 22.4,
    "curah_hujan_mm_per_day": 110,
    "pola_tanam": "polikultur"
  }
}
```

### 5. **POST /batch_predict/** - Batch Prediction

Prediksi untuk multiple data sekaligus

**Request Body:**

```json
{
  "data": [
    {
      "elevasi_mdpl": 850,
      "suhu_c": 24.5,
      "curah_hujan_mm_per_day": 200,
      "pola_tanam": "polikultur"
    },
    {
      "elevasi_mdpl": 820,
      "suhu_c": 25.2,
      "curah_hujan_mm_per_day": 180,
      "pola_tanam": "monokultur"
    }
  ]
}
```

**Response:**

```json
{
  "predictions": [
    {
      "index": 0,
      "input": {
        "elevasi_mdpl": 850,
        "suhu_c": 24.5,
        "curah_hujan_mm_per_day": 200,
        "pola_tanam": "polikultur"
      },
      "produktivitas_pred": 1875.5
    },
    {
      "index": 1,
      "input": {
        "elevasi_mdpl": 820,
        "suhu_c": 25.2,
        "curah_hujan_mm_per_day": 180,
        "pola_tanam": "monokultur"
      },
      "produktivitas_pred": 1520.3
    }
  ]
}
```

### 6. **POST /reload_model/** - Reload Model

Reload model dari file (berguna jika model di-update)

**Response:**

```json
{
  "message": "Model berhasil direload ✅",
  "status": "success"
}
```

---

## 💡 Contoh Penggunaan

### Menggunakan cURL

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "elevasi_mdpl": 836,
    "suhu_c": 22.4,
    "curah_hujan_mm_per_day": 110,
    "pola_tanam": "polikultur"
  }'
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "elevasi_mdpl": 850,
        "suhu_c": 24.5,
        "curah_hujan_mm_per_day": 200,
        "pola_tanam": "polikultur"
      },
      {
        "elevasi_mdpl": 820,
        "suhu_c": 25.2,
        "curah_hujan_mm_per_day": 180,
        "pola_tanam": "monokultur"
      }
    ]
  }'
```

### Menggunakan Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict/",
    json={
        "elevasi_mdpl": 836,
        "suhu_c": 22.4,
        "curah_hujan_mm_per_day": 110,
        "pola_tanam": "polikultur"
    }
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch_predict/",
    json={
        "data": [
            {
                "elevasi_mdpl": 850,
                "suhu_c": 24.5,
                "curah_hujan_mm_per_day": 200,
                "pola_tanam": "polikultur"
            },
            {
                "elevasi_mdpl": 820,
                "suhu_c": 25.2,
                "curah_hujan_mm_per_day": 180,
                "pola_tanam": "monokultur"
            }
        ]
    }
)
print(response.json())
```

### Menggunakan JavaScript/Fetch

```javascript
// Single prediction
fetch("http://localhost:8000/predict/", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    elevasi_mdpl: 836,
    suhu_c: 22.4,
    curah_hujan_mm_per_day: 110,
    pola_tanam: "polikultur",
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

---

## 🔄 Alur Preprocessing

API ini menggunakan **ALUR PREPROCESSING YANG SAMA PERSIS** dengan yang ada di Colab notebook:

### Step 1: One-Hot Encoding

```python
# Mengubah pola_tanam ('polikultur' atau 'monokultur') menjadi binary features
pola_encoded = onehot_encoder_pola.transform(df[['pola_tanam']])
```

### Step 2: Feature Combination

```python
# Menggabungkan fitur numerik dengan hasil encoding
X_numeric = df[['elevasi_mdpl', 'suhu_c', 'curah_hujan_mm_per_day']].values
X = np.hstack([X_numeric, pola_encoded])
```

### Step 3: Scaling

```python
# Normalisasi menggunakan StandardScaler (mean=0, std=1)
X_scaled = scaler.transform(X)
```

### Step 4: Prediction

```python
# Prediksi menggunakan XGBoost model
produktivitas_pred = model_produktivitas.predict(X_scaled)[0]
```

---

## 📊 Struktur Data Input

| Field                    | Type   | Required | Description                                      | Example      |
| ------------------------ | ------ | -------- | ------------------------------------------------ | ------------ |
| `elevasi_mdpl`           | float  | Yes      | Elevasi dalam meter di atas permukaan laut       | 836          |
| `suhu_c`                 | float  | Yes      | Suhu rata-rata dalam Celsius                     | 22.4         |
| `curah_hujan_mm_per_day` | float  | Yes      | Curah hujan dalam mm per hari                    | 110          |
| `pola_tanam`             | string | Yes      | Jenis pola tanam: 'polikultur' atau 'monokultur' | "polikultur" |

---

## ⚠️ Error Handling

### Invalid pola_tanam

```json
{
  "detail": "Nilai pola_tanam tidak valid. Gunakan salah satu: ['monokultur', 'polikultur']"
}
```

### Missing Fields

```json
{
  "detail": "Kolom elevasi_mdpl tidak ditemukan"
}
```

### Model Not Loaded

```json
{
  "detail": "Model belum dimuat. Gunakan endpoint /reload_model/ untuk memuat model."
}
```

---

## 🎯 Tips & Best Practices

1. **Validasi Input**: Pastikan semua field required terisi dengan tipe data yang benar
2. **Batch Processing**: Untuk prediksi multiple data, gunakan `/batch_predict/` untuk efisiensi
3. **Error Handling**: Selalu handle error response dari API
4. **Model Update**: Setelah re-training model, gunakan `/reload_model/` tanpa perlu restart API
5. **Documentation**: Gunakan `/docs` untuk testing interaktif

---

## 📝 File Structure

```
fast-api-xgboost-kopi/
├── main.py                 # FastAPI application
├── convert_model.py        # Script untuk convert/train model
├── test_api.py            # Script untuk test API
├── requirements.txt        # Python dependencies
├── test_sample2.csv       # Sample training data
├── saved/
│   └── model.pkl          # Trained model
└── README.md              # Documentation (file ini)
```

---

## 🤝 Support

Jika ada pertanyaan atau issue:

1. Check dokumentasi di `/docs`
2. Test dengan `/health/` untuk cek status
3. Review error message di response
4. Cek console output untuk detail error

---

**Version:** 1.0.0
**Last Updated:** December 2025
