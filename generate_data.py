import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000  # number of rows

speed = np.random.randint(20, 140, n)
hour = np.random.randint(0, 24, n)
weather = np.random.choice(["Clear", "Rain", "Fog", "Cloudy"], n, p=[0.5, 0.2, 0.1, 0.2])
road_type = np.random.choice(["Highway", "City", "Rural"], n)
traffic = np.random.choice(["Low", "Medium", "High"], n)
driver_age = np.random.randint(18, 75, n)

# Calculate a risk score using simple rules
score = np.zeros(n)
score += (speed > 80) * 2 + (speed > 110) * 2
score += ((hour >= 22) | (hour <= 5)) * 2
score += (weather == "Rain") * 2 + (weather == "Fog") * 3
score += (road_type == "Rural") * 1 + (road_type == "Highway") * 1
score += (traffic == "High") * 1
score += ((driver_age < 25) | (driver_age > 65)) * 1
score += np.random.randint(0, 3, n)  # some randomness

risk = np.where(score <= 3, "Low", np.where(score <= 6, "Medium", "High"))

df = pd.DataFrame({
    "speed": speed,
    "hour": hour,
    "weather": weather,
    "road_type": road_type,
    "traffic": traffic,
    "driver_age": driver_age,
    "risk": risk
})

df.to_csv("accident_data.csv", index=False)
print("Dataset created!")
print(df.head())
print(df["risk"].value_counts())