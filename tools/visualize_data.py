import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])

df["hour"] = df["LastUpdated"].dt.hour

hourly = df.groupby("hour")["Occupancy"].mean()

plt.plot(hourly.index, hourly.values)
plt.xlabel("Saat")
plt.ylabel("Ortalama Doluluk")
plt.title("Saatlere Gore Otopark Dolulugu")
plt.show()