import requests

with open("planets.csv", "rb") as f:
    files = {'file': f}
    res = requests.post("http://localhost:8000/upload", files=files)

print(res.json().get("columns"))


dropped_cols = ["kepid", "kepler_name", "kepoi_name", "koi_pdisposition", "koi_teq_err1", "koi_teq_err2", "koi_tce_delivname"]



data = {
            "has_new_data": False
            "dropped_cols": dropped_cols,
            "target_col": "koi_disposition",
            "split": 0.2,
            "n_estimators": 100,
            "learning_rate": 0.01,
            "random_state": 0,
            "max_iter": 3000,
            "max_iter_final": 3000
            }

r = requests.post("http://localhost:8000/train", json=data)
print(r.json())


print(scores)
