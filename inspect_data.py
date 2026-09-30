import pandas as pd

path = r"C:\Users\student\Downloads\US_Accidents_March23.csv"

df = pd.read_csv(path, nrows=1000)
print("Number of columns:", len(df.columns))
print()
print("Column names:")
for c in df.columns:
    print(" -", c)
print()
print(df.head(3).to_string())