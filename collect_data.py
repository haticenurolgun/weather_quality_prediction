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
weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,surface_pressure,visibility,dew_point_2m,temperature_80m,cloud_cover"
air_quality_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=pm10,pm2_5,carbon_dioxide,ammonia,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide"

print("Fetching data from API...")

# 3. Adım: API'den verilerin çekilmesi (JSON formatında)
weather_response = requests.get(weather_url).json()
air_quality_response = requests.get(air_quality_url).json()

if "error" in weather_response:
    print("HAVA DURUMU API HATASI:", weather_response["reason"])
if "error" in air_quality_response:
    print("HAVA KALİTESİ API HATASI:", air_quality_response["reason"])


# 5. Adım: Verilerin Pandas DF  yapısına dönüştürülmesi
df_weather = pd.DataFrame(weather_response['hourly'])
df_air_quality = pd.DataFrame(air_quality_response['hourly'])

# 6. Adım: Hava durumu ve kalite tablolarının 'time' (zaman) sütunu üzerinden birleştirilmesi
df_combined = pd.merge(df_weather, df_air_quality, on='time')


# Eksik verilerin temizlenmesi
df_combined = df_combined.dropna()

#saat bilgisi ekleme (typecasting)
df_combined['time'] = pd.to_datetime(df_combined['time'])
df_combined['hour'] = df_combined['time'].dt.hour

def get_traffic_density(hour):
    
    if 7 <= hour <= 9:
        return 2  # Yüksek Trafik (Sabah işe gidiş)
    elif 17 <= hour <= 19:
        return 2  # Yüksek Trafik (Akşam iş çıkışı)
    elif 10 <= hour <= 16:
        return 1  # Orta Trafik (Gün içi)
    else:
        return 0  # Düşük Trafik (Gece ve sabaha karşı)

df_combined['Traffic_Density_Proxy'] = df_combined['hour'].apply(get_traffic_density)

print(f"Data fetching successful! Total rows: {len(df_combined)}")

# 8. Adım: Hazırlanan veri setinin CSV olarak kaydedilmesi
df_combined.to_csv("ankara_weather_data.csv", index=False)
print("Data successfully saved as 'ankara_weather_data.csv'")

# 9. Adım: EDA 
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))


# --- GRAFİK 1: Rüzgar Hızı ve PM10 İlişkisi ---
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_combined, 
    x="wind_speed_10m",   
    y="pm10",            
    hue="temperature_2m", 
    palette="coolwarm", 
    s=80
)

plt.title("Ankara: Rüzgar Hızı ve PM10 Kirliliği İlişkisi", fontsize=14, fontweight="bold")
plt.xlabel("Rüzgar Hızı (km/h)", fontsize=12)
plt.ylabel("PM10 Kirlilik Seviyesi", fontsize=12)
plt.show()

#grafik 2:korelasyon ısı haritası

plt.figure(figsize=(14, 10))
correlation_data = df_combined.drop(columns=['time'])
correlation_matrix = correlation_data.corr()

sns.heatmap(
    correlation_matrix, 
    annot=True,          # Kutuların içine rakamları yaz
    fmt=".2f",           
    cmap="coolwarm",     
    linewidths=0.5
)

plt.title(" Korelasyon Matrisi", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, ha='right') # Alt yazıları yan yatırmak için
plt.tight_layout()
plt.show()

