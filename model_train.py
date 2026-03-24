import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Adım: Kaydedilen veri setinin okunması
print("Loading dataset...")
df = pd.read_csv("ankara_weather_data.csv")

# Özellikler (Girdiler) ve Hedef (Çıktı) değişkenlerinin belirlenmesi
X = df.drop(columns=['time', 'pm10', 'pm2_5'])
y = df['pm10']

# Verinin %80 Eğitim (Train) ve %20 Test olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training models...")


# --- RANDOM FOREST ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_error = mean_absolute_error(y_test, rf_predictions)

standart_values = {
    'temperature_2m': [15.0],
    'relative_humidity_2m': [40.0],
    'wind_speed_10m': [10.0],     # Rüzgar
    'wind_direction_10m': [180.0],
    'precipitation': [0.0],
    'surface_pressure': [900.0],
    'visibility': [10000.0],
    'dew_point_2m': [5.0],
    'temperature_80m': [13.0],
    'cloud_cover': [20.0],
    'ammonia': [0.5],
    'nitrogen_dioxide': [15.0],   # Egzoz Gazı
    'sulphur_dioxide': [2.0],
    'carbon_monoxide': [200.0],   # Egzoz Gazı
    'carbon_dioxide': [400.0],    
    'hour': [12],                 # Saat
    'Traffic_Density_Proxy': [1]  # Trafik
}

# 2. Adım: Canlı test senaryolarının oluşturulması
# Senaryo 1: Trafik saati
scenario_1 = pd.DataFrame(standart_values)
scenario_1['hour'] = 8
scenario_1['Traffic_Density_Proxy'] = 2
scenario_1['wind_speed_10m'] = 0.0          
scenario_1['carbon_monoxide'] = 800.0      
scenario_1['nitrogen_dioxide'] = 50.0       

# Senaryo 2: Trafik olmayan saat, fırtına var
scenario_2 = pd.DataFrame(standart_values)
scenario_2['hour'] = 3
scenario_2['Traffic_Density_Proxy'] = 0
scenario_2['wind_speed_10m'] = 35.0         
scenario_2['carbon_monoxide'] = 50.0        
scenario_2['nitrogen_dioxide'] = 5.0        

# --- SÜTUNLARI EĞİTİM SETİ İLE AYNI SIRAYA SOKMAK İÇİN ---
scenario_1 = scenario_1[X_train.columns]
scenario_2 = scenario_2[X_train.columns]


# 3. Adım: Model sonuçlarının ve tahminlerinin karşılaştırmalı bir tabloya dönüştürülmesi
results = {
    "Model_Name": ["Random Forest"],
    "MAE_Error (Hata)": [round(rf_error, 2)],
    "PREDİCTİON_SCENARİO_1": [ round(rf_model.predict(scenario_1)[0], 2)],
    "PREDİCTİON_SCENARİO_2": [round(rf_model.predict(scenario_2)[0], 2)]
}

df_results = pd.DataFrame(results)

# Sonuçların ekrana yazdırılması
print("\n--- [MODEL COMPARISON RESULTS] ---")
print(df_results.to_string(index=False))

# Karşılaştırma tablosunun CSV olarak kaydedilmesi
df_results.to_csv("model_comparison_results.csv", index=False)
print("\n Results successfully saved to 'model_comparison_results.csv'!")

# 4. Adım: Eğitilen modelleri kaydetme
joblib.dump(rf_model, "random_forest_weather_model_v2.joblib")

print("\n Model successfully saved!")
