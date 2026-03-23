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
X = df[['Temperature_C', 'Humidity_Percent', 'Wind_Speed_kmh']]
y = df['PM10_Pollution']

# Verinin %80 Eğitim (Train) ve %20 Test olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training models...")

# --- MODEL 1: LINEAR REGRESSION (DOĞRUSAL REGRESYON - BASELINE) ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_error = mean_absolute_error(y_test, lr_predictions)

# --- MODEL 2: RANDOM FOREST ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_error = mean_absolute_error(y_test, rf_predictions)

# 2. Adım: Canlı test senaryolarının oluşturulması
# Senaryo 1: Rüzgarsız hava (Wind Speed = 0)
scenario_calm = pd.DataFrame({'Temperature_C': [15], 'Humidity_Percent': [40], 'Wind_Speed_kmh': [0]})

# Senaryo 2: Fırtınalı hava (Wind Speed = 30)
scenario_stormy = pd.DataFrame({'Temperature_C': [15], 'Humidity_Percent': [40], 'Wind_Speed_kmh': [30]})

# 3. Adım: Model sonuçlarının ve tahminlerinin karşılaştırmalı bir tabloya dönüştürülmesi
results = {
    "Model_Name": ["Linear Regression", "Random Forest"],
    "MAE_Error": [round(lr_error, 2), round(rf_error, 2)],
    "Prediction_Calm_Weather": [round(lr_model.predict(scenario_calm)[0], 2), round(rf_model.predict(scenario_calm)[0], 2)],
    "Prediction_Stormy_Weather": [round(lr_model.predict(scenario_stormy)[0], 2), round(rf_model.predict(scenario_stormy)[0], 2)]
}

df_results = pd.DataFrame(results)

# Sonuçların ekrana yazdırılması
print("\n--- [MODEL COMPARISON RESULTS] ---")
print(df_results.to_string(index=False))

# Karşılaştırma tablosunun CSV olarak kaydedilmesi
df_results.to_csv("model_comparison_results.csv", index=False)
print("\n Results successfully saved to 'model_comparison_results.csv'!")

# 4. Adım: Eğitilen modelleri kaydetme
joblib.dump(rf_model, "random_forest_weather_model.joblib")
joblib.dump(lr_model, "linear_regression_weather_model.joblib")

print("\n Models successfully saved!")
print(" -> random_forest_weather_model.joblib")
print(" -> linear_regression_weather_model.joblib")