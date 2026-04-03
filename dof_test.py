import pandas as pd

df = pd.read_parquet("/home/ycn/hilserl_sim/lerobot/src/lerobot/ycn_hilserl_sim/ycn_records_keyboard/data/chunk-000/file-000.parquet")
print(df["action"].iloc[0])
print(len(df["action"].iloc[0]))
for i in range(500):
    print(df["action"].iloc[i])
