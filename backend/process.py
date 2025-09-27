import pandas as pd

df = pd.read_csv("planets.csv")
dropped_cols = ["kepid", "kepoi_name", "kepler_name", "koi_pdisposition",
                "koi_teq_err1", "koi_teq_err2", "koi_tce_delivname"]

X = df.drop(dropped_cols, axis=1)
# X = X.drop("koi_disposition", axis=1)
X.dropna()

last = X.tail(10)
print(last)
last.to_csv("testing.csv", index=False)
