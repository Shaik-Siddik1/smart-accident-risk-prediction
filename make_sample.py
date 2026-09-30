import pandas as pd

path = r"C:\Users\student\Downloads\US_Accidents_March23.csv"

cols = ["Severity", "Start_Time", "Temperature(F)", "Humidity(%)",
        "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
        "Weather_Condition", "Junction", "Crossing", "Traffic_Signal",
        "Stop", "Railway", "Sunrise_Sunset"]


def group_weather(w):
    w = str(w).lower()
    if any(k in w for k in ["rain", "drizzle", "shower", "thunder"]):
        return "Rain"
    if any(k in w for k in ["snow", "sleet", "ice", "hail"]):
        return "Snow"
    if any(k in w for k in ["fog", "mist", "haze", "smoke"]):
        return "Fog"
    if any(k in w for k in ["cloud", "overcast"]):
        return "Cloudy"
    if any(k in w for k in ["fair", "clear"]):
        return "Clear"
    return "Other"


parts = []
total = 0
for chunk in pd.read_csv(path, usecols=cols, chunksize=500000):
    total += len(chunk)
    print("Rows read so far:", total)

    # keep a random part of each severity, so the file stays small
    keep = [
        chunk[chunk.Severity == 1],
        chunk[chunk.Severity == 2].sample(frac=0.05, random_state=1),
        chunk[chunk.Severity == 3].sample(frac=0.10, random_state=1),
        chunk[chunk.Severity == 4],
    ]
    parts.append(pd.concat(keep))

df = pd.concat(parts)

# Clean the data
df = df.dropna(subset=["Temperature(F)", "Humidity(%)", "Visibility(mi)",
                       "Wind_Speed(mph)", "Weather_Condition",
                       "Sunrise_Sunset", "Start_Time"])
df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0)

# New columns
df["hour"] = df["Start_Time"].str[11:13].astype(int)
df["month"] = df["Start_Time"].str[5:7].astype(int)
df["weather"] = df["Weather_Condition"].apply(group_weather)
df["is_night"] = (df["Sunrise_Sunset"] == "Night").astype(int)
for c in ["Junction", "Crossing", "Traffic_Signal", "Stop", "Railway"]:
    df[c] = df[c].astype(int)

# Risk label from Severity
df["risk"] = df["Severity"].map({1: "Low", 2: "Medium", 3: "High", 4: "High"})

df = df.rename(columns={
    "Temperature(F)": "temperature", "Humidity(%)": "humidity",
    "Visibility(mi)": "visibility", "Wind_Speed(mph)": "wind_speed",
    "Precipitation(in)": "precipitation", "Junction": "junction",
    "Crossing": "crossing", "Traffic_Signal": "traffic_signal",
    "Stop": "stop", "Railway": "railway"})

final_cols = ["hour", "month", "temperature", "humidity", "visibility",
              "wind_speed", "precipitation", "weather", "is_night",
              "junction", "crossing", "traffic_signal", "stop", "railway",
              "risk"]
df = df[final_cols]

# Balance: same number of Low, Medium, High
n = df["risk"].value_counts().min()
n = min(n, 30000)
df = df.groupby("risk").sample(n=n, random_state=42)

df.to_csv("us_sample.csv", index=False)
print()
print("DONE! Saved us_sample.csv with", len(df), "rows")
print(df["risk"].value_counts())
print(df.head())