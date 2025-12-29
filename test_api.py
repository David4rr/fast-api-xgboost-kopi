# Test API Script
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

print("="*60)
print("🧪 TESTING PRODUCTIVITY PREDICTOR API")
print("="*60)

# Test 1: Health Check
print("\n1. Testing /health/ endpoint...")
response = requests.get(f"{BASE_URL}/health/")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2: Model Info
print("\n2. Testing /model_info/ endpoint...")
response = requests.get(f"{BASE_URL}/model_info/")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 3: Single Prediction - Polikultur
print("\n3. Testing /predict/ endpoint - Polikultur...")
test_data = {
    "elevasi_mdpl": 836,
    "suhu_c": 22.4,
    "curah_hujan_mm_per_day": 110,
    "pola_tanam": "polikultur"
}
response = requests.post(f"{BASE_URL}/predict/", json=test_data)
print(f"Status: {response.status_code}")
print(f"Input: {test_data}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 4: Single Prediction - Monokultur
print("\n4. Testing /predict/ endpoint - Monokultur...")
test_data2 = {
    "elevasi_mdpl": 800,
    "suhu_c": 26.0,
    "curah_hujan_mm_per_day": 160,
    "pola_tanam": "monokultur"
}
response = requests.post(f"{BASE_URL}/predict/", json=test_data2)
print(f"Status: {response.status_code}")
print(f"Input: {test_data2}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 5: Batch Prediction
print("\n5. Testing /batch_predict/ endpoint...")
batch_data = {
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
        },
        {
            "elevasi_mdpl": 880,
            "suhu_c": 23.8,
            "curah_hujan_mm_per_day": 220,
            "pola_tanam": "polikultur"
        }
    ]
}
response = requests.post(f"{BASE_URL}/batch_predict/", json=batch_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 6: Error handling - Invalid pola_tanam
print("\n6. Testing error handling - Invalid pola_tanam...")
invalid_data = {
    "elevasi_mdpl": 850,
    "suhu_c": 24.5,
    "curah_hujan_mm_per_day": 200,
    "pola_tanam": "invalid_type"
}
response = requests.post(f"{BASE_URL}/predict/", json=invalid_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

print("\n" + "="*60)
print("✅ TESTING SELESAI!")
print("="*60)
