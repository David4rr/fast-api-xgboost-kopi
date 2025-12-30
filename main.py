# fastapi_app.py
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uvicorn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

class PredictionInput(BaseModel):
    elevasi_mdpl: float
    suhu_c: float
    curah_hujan_mm_per_day: float
    pola_tanam: str = Field(..., description="Jenis pola tanam: 'polikultur' atau 'monokultur'")

class PredictionResponse(BaseModel):
    produktivitas_pred: float
    input_data: Dict

class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict]

app = FastAPI(
    title="Productivity Predictor API",
    description="API untuk prediksi produktivitas pertanian menggunakan model XGBoost",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_produktivitas = None
scaler = None
onehot_encoder_pola = None
is_loaded = False

def load_model_components():
    """Load model dan komponen preprocessing."""
    global model_produktivitas, scaler, onehot_encoder_pola, is_loaded
    
    try:
        # Path ke file model yang disave dari Colab
        model_path = 'saved/model.pkl'
        
        if not os.path.exists(model_path):
            print(f"File model tidak ditemukan di: {model_path}")
            print("Pastikan file model.pkl dari Colab sudah ada di folder 'saved/'")
            return False
        
        # Load semua komponen dari file pickle
        print("Loading model components...")
        loaded = joblib.load(model_path)
        
        # Extract komponen sesuai struktur di Colab
        model_produktivitas = loaded['model_produktivitas']
        scaler = loaded['scaler']
        onehot_encoder_pola = loaded['onehot_encoder_pola']
        print("Model produktivitas loaded")
        print("Scaler loaded")
        print("OneHotEncoder pola_tanam loaded")
        # Validasi bahwa semua komponen sudah fitted
        if not hasattr(scaler, 'mean_'):
            print("Warning: Scaler belum fitted!")
            return False
            
        if not hasattr(onehot_encoder_pola, 'categories_'):
            print("Warning: OneHotEncoder belum fitted!")
            return False
        
        print(f"Kategori pola_tanam: {list(onehot_encoder_pola.categories_[0])}")
        print(f"Jumlah fitur model: {model_produktivitas.n_features_in_}")
        is_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    """Load model saat aplikasi startup."""
    global is_loaded
    success = load_model_components()
    if success:
        print("API siap digunakan")
    else:
        print("API berjalan tapi model belum loaded. Gunakan /reload_model/ untuk load ulang.")

@app.get("/")
async def root():
    return {
        "message": "Productivity Predictor API",
        "status": "Model loaded" if is_loaded else "Model not loaded",
        "endpoints": {
            "predict": "/predict/ - Prediksi tunggal",
            "batch_predict": "/batch_predict/ - Prediksi batch",
            "health": "/health/ - Cek status API",
            "reload_model": "/reload_model/ - Reload model",
            "model_info": "/model_info/ - Info tentang model"
        }
    }

@app.get("/health/")
async def health_check():
    """Endpoint untuk mengecek kesehatan API"""
    return {
        "status": "healthy",
        "model_loaded": is_loaded,
        "model_type": "XGBoost Regressor" if model_produktivitas else None,
        "preprocessing": {
            "scaler": "StandardScaler" if scaler else None,
            "encoder": "OneHotEncoder" if onehot_encoder_pola else None
        }
    }

@app.post("/reload_model/")
async def reload_model():
    """Reload model dari file"""
    success = load_model_components()
    if success:
        return {"message": "Model berhasil direload", "status": "success"}
    else:
        raise HTTPException(status_code=500, detail="Gagal memuat model. Cek console untuk detail error.")

def preprocess_input(df):
    """
    Preprocessing SAMA PERSIS seperti di Colab
    Mengikuti alur: OneHotEncoding → Combine → Scaling
    """
    global onehot_encoder_pola, scaler
    # One Hot Encoding untuk pola_tanam
    pola_encoded = onehot_encoder_pola.transform(df[['pola_tanam']])
    # Gabungkan fitur numerik dengan hasil encoding
    X_numeric = df[['elevasi_mdpl', 'suhu_c', 'curah_hujan_mm_per_day']].values
    X = np.hstack([X_numeric, pola_encoded])
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

@app.post("/predict/", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """
    Endpoint untuk prediksi produktivitas tunggal
    ALUR PREPROCESSING SAMA SEPERTI DI COLAB
    
    Contoh input:
    ```json
    {
        "elevasi_mdpl": 836,
        "suhu_c": 22.4,
        "curah_hujan_mm_per_day": 110,
        "pola_tanam": "polikultur"
    }
    ```
    """
    if not is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model belum dimuat. Gunakan endpoint /reload_model/ untuk memuat model."
        )
    
    try:
        # Convert input ke DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Validasi kolom yang diperlukan
        required_columns = ['elevasi_mdpl', 'suhu_c', 'curah_hujan_mm_per_day', 'pola_tanam']
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Kolom {col} tidak ditemukan")
        
        # Validasi nilai pola_tanam
        valid_pola = list(onehot_encoder_pola.categories_[0])
        if input_dict['pola_tanam'] not in valid_pola:
            raise HTTPException(
                status_code=400, 
                detail=f"Nilai pola_tanam tidak valid. Gunakan salah satu: {valid_pola}"
            )
        
        # PREPROCESSING
        X_scaled = preprocess_input(df)
        
        # PREDICTION
        produktivitas_pred = model_produktivitas.predict(X_scaled)[0]
        
        return {
            "produktivitas_pred": float(produktivitas_pred),
            "input_data": input_dict
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dalam prediksi: {str(e)}")

@app.post("/batch_predict/", response_model=BatchPredictionResponse)
async def batch_predict(batch_input: BatchPredictionInput):
    """
    Endpoint untuk prediksi batch
    ALUR PREPROCESSING SAMA SEPERTI DI COLAB
    
    Contoh input:
    ```json
    {
        "data": [
            {
                "elevasi_mdpl": 836,
                "suhu_c": 22.4,
                "curah_hujan_mm_per_day": 110,
                "pola_tanam": "polikultur"
            },
            {
                "elevasi_mdpl": 860,
                "suhu_c": 23.8,
                "curah_hujan_mm_per_day": 180,
                "pola_tanam": "monokultur"
            }
        ]
    }
    ```
    """
    if not is_loaded:
        raise HTTPException(status_code=503, detail="Model belum dimuat. Gunakan endpoint /reload_model/ untuk memuat model.")
    
    try:
        # Convert batch input ke DataFrame
        data_list = [item.dict() for item in batch_input.data]
        df = pd.DataFrame(data_list)
        
        # Validasi kolom
        required_columns = ['elevasi_mdpl', 'suhu_c', 'curah_hujan_mm_per_day', 'pola_tanam']
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Kolom {col} tidak ditemukan")
        
        # Validasi semua nilai pola_tanam
        valid_pola = list(onehot_encoder_pola.categories_[0])
        invalid_pola = set(df['pola_tanam'].unique()) - set(valid_pola)
        if invalid_pola:
            raise HTTPException(
                status_code=400, 
                detail=f"Nilai pola_tanam tidak valid: {invalid_pola}. Gunakan: {valid_pola}"
            )
        
        # PREPROCESSING
        X_scaled = preprocess_input(df)
        
        # PREDICTIONS
        predictions = model_produktivitas.predict(X_scaled)
        
        # Format response
        results = []
        for i, (input_item, pred) in enumerate(zip(data_list, predictions)):
            results.append({
                "index": i,
                "input": input_item,
                "produktivitas_pred": float(pred)
            })
        
        return {"predictions": results}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dalam batch prediksi: {str(e)}")

# Endpoint untuk mendapatkan informasi model
@app.get("/model_info/")
async def get_model_info():
    """Mendapatkan informasi tentang model yang digunakan"""
    if not is_loaded:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    
    try:
        # Get feature importance
        feature_importance = None
        if hasattr(model_produktivitas, 'feature_importances_'):
            # Nama fitur SAMA SEPERTI DI COLAB
            base_features = ['elevasi_mdpl', 'suhu_c', 'curah_hujan_mm_per_day']
            pola_categories = onehot_encoder_pola.categories_[0]
            pola_feature_names = [f'pola_{cat}' for cat in pola_categories]
            feature_names = base_features + pola_feature_names
            
            feature_importance = {
                name: float(imp) 
                for name, imp in zip(feature_names, model_produktivitas.feature_importances_)
            }
            
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        return {
            "model_type": type(model_produktivitas).__name__,
            "is_fitted": hasattr(model_produktivitas, 'feature_importances_'),
            "n_features": model_produktivitas.n_features_in_ if hasattr(model_produktivitas, 'n_features_in_') else None,
            "feature_importance": feature_importance,
            "pola_tanam_categories": list(onehot_encoder_pola.categories_[0]),
            "preprocessing_pipeline": {
                "step_1": "OneHotEncoding untuk pola_tanam",
                "step_2": "Combine numeric features dengan encoded pola_tanam",
                "step_3": "StandardScaler untuk normalisasi"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mendapatkan info model: {str(e)}")

if __name__ == "__main__":
    print("="*60)
    print("Starting Productivity Predictor API")
    print("="*60)
    print("Pastikan file model ada di: saved/model.pkl")
    print("API akan berjalan di: http://0.0.0.0:8000")
    print("Dokumentasi API: http://0.0.0.0:8000/docs")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8001)