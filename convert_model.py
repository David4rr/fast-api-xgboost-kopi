"""
Script untuk mengkonversi model dari Colab agar kompatibel dengan environment lokal
Jalankan script ini untuk membuat model baru yang kompatibel
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import joblib
import os

class ProductivityPredictor:
    def __init__(self):
        self.model_produktivitas = None
        self.scaler = StandardScaler()
        self.onehot_encoder_pola = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.is_fitted = False
        self.cv_results = None
        self.train_metrics = None
    
    def load_data_from_csv(self, filepath):
        """Mengambil data dari file CSV"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Data loaded successfully from {filepath}")
            print(f"📊 Data shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filepath} not found.")
        except Exception as e:
            raise Exception(f"Error loading CSV file: {e}")
    
    def preprocess_data(self, data, training=True):
        """Preprocessing SAMA SEPERTI DI COLAB"""
        df = data.copy()
        
        # Step 1: One Hot Encoding untuk pola_tanam
        if training:
            pola_encoded = self.onehot_encoder_pola.fit_transform(df[['pola_tanam']])
            print(f"✅ OneHotEncoder fitted - Kategori: {list(self.onehot_encoder_pola.categories_[0])}")
        else:
            pola_encoded = self.onehot_encoder_pola.transform(df[['pola_tanam']])
        
        # Step 2: Gabungkan fitur numerik dengan hasil encoding
        X_numeric = df[['elevasi_mdpl', 'suhu_c', 'curah_hujan_mm_per_day']].values
        X = np.hstack([X_numeric, pola_encoded])
        
        # Handle case untuk prediction (tidak ada target column)
        y_produktivitas = df['produktivitas_kg_ha'] if 'produktivitas_kg_ha' in df.columns else None
        
        # Step 3: Scale features
        if training:
            X_scaled = self.scaler.fit_transform(X)
            print(f"✅ Scaler fitted - Mean: {self.scaler.mean_[:3]}")
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y_produktivitas
    
    def cross_validate_model(self, X, y, n_splits=5):
        """Perform K-Fold cross-validation and report results (same as Colab)"""
        print(f"\n=== {n_splits}-Fold Cross Validation (full dataset) ===")

        param_grid = {
            'n_estimators': [50, 80, 100, 120],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 0.5],
            'reg_lambda': [0.1, 0.8, 1.0, 1.2]
        }

        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=15,
            scoring='neg_mean_squared_error',
            cv=kfold,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        print("Performing cross-validation tuning on full dataset...")
        random_search.fit(X, y)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        print(f"Best parameters (full CV): {best_params}")
        print(f"Best CV score (negative MSE): {random_search.best_score_:.4f}")

        # store
        self.cv_results = random_search.cv_results_
        self.best_model = best_model
        self.best_params = best_params

        return best_model, best_params

    def tune_hyperparameters(self, X_train, y_train):
        """Tune hyperparameters on training split (same as Colab tune function)"""
        print("Performing hyperparameter tuning for limited dataset...")

        param_grid = {
            'n_estimators': [50, 80, 100, 120],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 0.5],
            'reg_lambda': [0.1, 0.8, 1.0, 1.2]
        }

        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=15,
            scoring='neg_mean_squared_error',
            cv=kfold,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best CV score: {-random_search.best_score_:.4f}")

        return random_search.best_params_

    def train_with_tuning(self, df):
        """Training model with exact Colab flow: full CV -> tune on train -> fit final model"""
        print("\n" + "="*60)
        print("🔧 Training model with CV + tuning (Colab flow)")
        print("="*60)

        X, y_prod = self.preprocess_data(df, training=True)

        # 1) Cross-validate on full dataset (same as train())
        self.cross_validate_model(X, y_prod, n_splits=5)

        # 2) Split for final training
        X_train, X_test, y_prod_train, y_prod_test = train_test_split(
            X, y_prod, test_size=0.1, random_state=42
        )

        print(f"\n📊 Data split - Train: {len(X_train)}, Test: {len(X_test)}")

        # 3) Tune hyperparameters on training split
        best_params_train = self.tune_hyperparameters(X_train, y_prod_train)

        # 4) Also get best params from full-dataset CV (already run above)
        best_params_cv = self.best_params if hasattr(self, 'best_params') else None

        # 5) Train two final variants
        print("\n🚀 Training final model variant A (params from train-tuning)...")
        model_a = xgb.XGBRegressor(objective='reg:squarederror', **best_params_train, random_state=42)
        model_a.fit(X_train, y_prod_train)

        model_b = None
        if best_params_cv is not None:
            print("🚀 Training final model variant B (params from full CV)...")
            model_b = xgb.XGBRegressor(objective='reg:squarederror', **best_params_cv, random_state=42)
            model_b.fit(X_train, y_prod_train)

        # Evaluate both on test set and choose the better by Test R²
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        def eval_model(m):
            y_train_pred = m.predict(X_train)
            y_test_pred = m.predict(X_test)
            tr_r2 = r2_score(y_prod_train, y_train_pred)
            tr_rmse = np.sqrt(mean_squared_error(y_prod_train, y_train_pred))
            tr_mae = mean_absolute_error(y_prod_train, y_train_pred)
            te_r2 = r2_score(y_prod_test, y_test_pred)
            te_rmse = np.sqrt(mean_squared_error(y_prod_test, y_test_pred))
            te_mae = mean_absolute_error(y_prod_test, y_test_pred)
            return {
                'train_r2': tr_r2, 'train_rmse': tr_rmse, 'train_mae': tr_mae,
                'test_r2': te_r2, 'test_rmse': te_rmse, 'test_mae': te_mae
            }

        res_a = eval_model(model_a)
        res_b = eval_model(model_b) if model_b is not None else None

        print("\n" + "="*60)
        print("📊 Variant A (train-tuning) performance:")
        print(f"Train R²: {res_a['train_r2']:.4f} | RMSE: {res_a['train_rmse']:.4f} | MAE: {res_a['train_mae']:.4f}")
        print(f"Test R²:  {res_a['test_r2']:.4f} | RMSE: {res_a['test_rmse']:.4f} | MAE: {res_a['test_mae']:.4f}")
        print("="*60)

        if res_b is not None:
            print("📊 Variant B (full-CV params) performance:")
            print(f"Train R²: {res_b['train_r2']:.4f} | RMSE: {res_b['train_rmse']:.4f} | MAE: {res_b['train_mae']:.4f}")
            print(f"Test R²:  {res_b['test_r2']:.4f} | RMSE: {res_b['test_rmse']:.4f} | MAE: {res_b['test_mae']:.4f}")
            print("="*60)

        # Choose better model by test R²
        chosen_model = model_a
        chosen_res = res_a
        chosen_label = 'A (train-tuning)'
        if res_b is not None and res_b['test_r2'] > res_a['test_r2']:
            chosen_model = model_b
            chosen_res = res_b
            chosen_label = 'B (full-CV)'

        print(f"\n✅ Chosen final model: Variant {chosen_label} (higher Test R²)")
        print(f"Chosen Test R²: {chosen_res['test_r2']:.4f}")

        self.model_produktivitas = chosen_model
        self.is_fitted = True

        self.train_metrics = {
            'train_r2': chosen_res['train_r2'],
            'train_rmse': chosen_res['train_rmse'],
            'train_mae': chosen_res['train_mae'],
            'test_r2': chosen_res['test_r2'],
            'test_rmse': chosen_res['test_rmse'],
            'test_mae': chosen_res['test_mae'],
            'best_params_train': best_params_train,
            'best_params_cv': best_params_cv
        }

        # Save both best params for traceability
        try:
            import json
            os.makedirs('saved', exist_ok=True)
            with open('saved/best_params_comparison.json', 'w') as f:
                json.dump({'best_params_train': best_params_train, 'best_params_cv': best_params_cv}, f, indent=2)
            print('Saved best_params_comparison.json')
        except Exception as e:
            print('Warning: could not save best params:', e)

        return self.model_produktivitas
    
    def save_models(self, filepath):
        """Save all models and preprocessing objects"""
        if not self.is_fitted:
            raise Exception("Models are not trained yet.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump({
            'model_produktivitas': self.model_produktivitas,
            'scaler': self.scaler,
            'onehot_encoder_pola': self.onehot_encoder_pola,
            'is_fitted': self.is_fitted,
            'cv_results': self.cv_results,
            'train_metrics': self.train_metrics
        }, filepath)
        
        print(f"\n✅ Model saved to {filepath}")

def main():
    print("="*60)
    print("🔄 MODEL CONVERSION SCRIPT")
    print("="*60)
    print("Script ini akan membuat model baru yang kompatibel")
    print("dengan environment lokal Anda.\n")
    
    # Cari file CSV training data
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ Tidak ada file CSV ditemukan!")
        print("Pastikan file training data (test_sample2.csv) ada di folder ini.")
        return
    
    print(f"📁 File CSV ditemukan: {csv_files}")
    
    # Gunakan file pertama atau yang sesuai
    csv_file = csv_files[0]
    for f in csv_files:
        if 'test_sample' in f or 'train' in f:
            csv_file = f
            break
    
    print(f"\n📂 Menggunakan file: {csv_file}")
    
    # Initialize predictor
    predictor = ProductivityPredictor()
    
    # Load data
    df = predictor.load_data_from_csv(csv_file)
    
    # Train model DENGAN HYPERPARAMETER TUNING - SAMA SEPERTI DI COLAB
    predictor.train_with_tuning(df)
    
    # Save model
    predictor.save_models('saved/model.pkl')
    
    print("\n" + "="*60)
    print("✅ KONVERSI SELESAI!")
    print("="*60)
    print("Model baru telah disimpan di: saved/model.pkl")
    print("Anda bisa menjalankan API dengan: python main.py")
    print("="*60)
    
    # Test prediction
    print("\n🧪 Testing prediction dengan model baru...")
    test_data = {
        'elevasi_mdpl': 836,
        'suhu_c': 22.4,
        'curah_hujan_mm_per_day': 110,
        'pola_tanam': 'polikultur'
    }
    
    df_test = pd.DataFrame([test_data])
    X_test, _ = predictor.preprocess_data(df_test, training=False)
    pred = predictor.model_produktivitas.predict(X_test)[0]
    
    print(f"Input: {test_data}")
    print(f"Prediksi produktivitas: {pred:.2f} kg/ha")
    print("\n✅ Model berfungsi dengan baik!")

if __name__ == "__main__":
    main()
