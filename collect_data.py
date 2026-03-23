import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Adım: Koordinatlar ve Tarih Aralığı (Ankara)
latitude = 39.9208
longitude = 32.8541
start_date = "2026-03-01"
end_date = "2026-03-23"

# 2. Adım: Veri çekilecek API URL'lerinin hazırlanması
weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
air_quality_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=pm10,pm2_5"

print("Fetching data from API...")

# 3. Adım: API'den verilerin çekilmesi (JSON formatında)
weather_response = requests.get(weather_url).json()
air_quality_response = requests.get(air_quality_url).json()

# 4. Adım: Saatlik verilerin listelerden ayrıştırılması
hourly_weather = weather_response['hourly']
hourly_air_quality = air_quality_response['hourly']

# 5. Adım: Verilerin Pandas DataFrame (tablo) yapısına dönüştürülmesi
df_weather = pd.DataFrame(hourly_weather)
df_air_quality = pd.DataFrame(hourly_air_quality)

# 6. Adım: Hava durumu ve kalite tablolarının 'time' (zaman) sütunu üzerinden birleştirilmesi
df_combined = pd.merge(df_weather, df_air_quality, on='time')

# 7. Adım: Sütun isimlerinin İngilizce ve model için temiz bir formata getirilmesi
df_combined.columns = ['Time', 'Temperature_C', 'Humidity_Percent', 'Wind_Speed_kmh', 'PM10_Pollution', 'PM25_Pollution']

# Eksik verilerin (NaN) temizlenmesi
df_combined = df_combined.dropna()

print(f"Data fetching successful! Total rows: {len(df_combined)}")

# 8. Adım: Hazırlanan veri setinin CSV olarak kaydedilmesi
df_combined.to_csv("ankara_weather_data.csv", index=False)
print("Data successfully saved as 'ankara_weather_data.csv'")

# 9. Adım: Veri Görselleştirme (EDA - Keşifçi Veri Analizi)
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))

# Rüzgar hızı ve hava kirliliği ilişkisinin dağılım grafiği (Scatter Plot)
sns.scatterplot(
    data=df_combined, 
    x="Wind_Speed_kmh", 
    y="PM10_Pollution", 
    hue="Temperature_C", 
    palette="coolwarm", 
    s=80
)

plt.title("Ankara (March 2026): Wind Speed vs Air Pollution", fontsize=14, fontweight="bold")
plt.xlabel("Wind Speed (km/h)", fontsize=12)
plt.ylabel("PM10 Pollution Level", fontsize=12)
plt.show()