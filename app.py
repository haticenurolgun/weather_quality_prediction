import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from collect_data import get_data
from datetime import datetime


class AnkaraWeatherApp:
    def __init__(self,root):
        self.root = root
        self.root.title("Air Quality Adviser")
        self.root.geometry("400x300")

        self.model=joblib.load("random_forest_weather_model_v2.joblib")

        self.root.configure(bg="pink")

        self.label = tk.Label(root, text="Check the Weather in Ankara", font=("Arial", 15))
        self.label.pack(pady=20)

        self.btn=tk.Button(root, text="Analyse for now.",command=self.analyse,bg="blue",fg="black",font=("Arial",12))
        self.btn.pack(pady=20)

        self.result_label=tk.Label(root,text="",font=("Arial", 11))
        self.result_label.pack(pady=20)

    def analyse(self):

        df_today= get_data(days_back=0)

        self.now=datetime.now()
        hour=self.now.hour
        needed_row=df_today[df_today['hour'] == hour]
        needed_row=needed_row.drop(columns=['time', 'pm10', 'pm2_5'])
        needed_row = needed_row[self.model.feature_names_in_] # modeldeki sıraya göre sıralamak için

        result= self.model.predict(needed_row)[0]
        
        if result<40:
            advice = "Air is clean, you should go walk!"
            color ="green"

        elif result <80:
            advice= "air is standart"
            color ="orange"

        else :
            advice="Air pollution is too high. Better to stay home."
            color = "red"
            
        self.result_label.config(text=f"Prediction: {result:.2f} (PM10)\n{advice}", fg=color)
            
if __name__ == "__main__":
    root = tk.Tk()
    app = AnkaraWeatherApp(root)
    root.mainloop() # Pencereyi ekranda tutan komut




