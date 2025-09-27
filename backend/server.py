from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import io
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

dataframes = {}
model_accuracy = None
model_recall = None
model_precision = None
cm = None
score = None
new_frames = {}
new_model_accuracy = None
new_model_recall = None
new_model_precision = None
new_cm = None
new_score = None


class Request(BaseModel):
    has_new_data: bool
    dropped_cols: List[str]
    target_col: str
    split: float
    n_estimators: int
    learning_rate: float
    random_state: int
    max_iter: int
    max_iter_final: int
    model_name: str


@app.get("/")
def hello():
    return {"Hello World!"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    data = pd.read_csv(io.StringIO(content.decode("utf-8")))
    dataframes["file"] = data

    sample = data.head().replace(
        [np.inf, -np.inf, np.nan], None).to_dict(orient="records")
    sample = jsonable_encoder(sample)

    return {
        "columns": data.columns.tolist(),
        "sample": sample,
    }


@app.get("/default")
async def get_default():
    pass


@app.post("/update")
async def append_data(file: UploadFile = File(...)):
    content = await file.read()
    new_data = pd.read_csv(io.StringIO(content.decode("utf-8")))
    copy = dataframes["file"]
    df = pd.concat(copy, new_data, ignore_index=True)
    df = df.drop_duplicates()

    new_frames["file"] = df

    sample = df.head().replace(
        [np.inf, -np.inf, np.nan], None).to_dict(orient="records")
    sample = jsonable_encoder(sample)

    return {
        "columns": df.columns.tolist(),
        "sample": sample,
    }


@app.post("/predict_default")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    data = pd.read_csv(io.StringIO(content.decode("utf-8")))
    model = joblib.load("default_model.joblib")
    prediction = model.predict(data)
    result = []
    for pred in prediction:
        if pred == 2:
            result.append('FALSE POSITITVE')
        if pred == 0:
            result.append("CANDIDATE")
        if pred == 1:
            result.append("CONFIRMED EXOPLANET")
    print(result)
    return {"result": result}


@app.post("/train")
async def train(req: Request):
    df = None
    if req.has_new_data:
        df = new_frames["file"]
    else:
        df = dataframes["file"]

    X = df.copy()
    X = X.drop(req.dropped_cols, axis=1)
    X = X.drop(req.target_col, axis=1)
    print(X)
    X = X.dropna(axis=1, how='all')
    print(X)
    X = X.dropna()
    print(X)
    y = df[req.target_col]
    y = y.loc[X.index]

    print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.split)

    abc = AdaBoostClassifier(n_estimators=req.n_estimators,
                             learning_rate=req.learning_rate, random_state=req.random_state)
    lr = LogisticRegression(max_iter=req.max_iter)
    rf = RandomForestClassifier()

    stack = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("abc", abc)],
        final_estimator=LogisticRegression(
            max_iter=req.max_iter_final, multi_class="multinomial"),
        stack_method="predict_proba",
        cv=5,
        passthrough=False
    )

    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro"),
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cross = cross_validate(stack, X_train, y_train,
                           cv=cv, scoring=scoring, n_jobs=-1)

    new_stack = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("abc", abc)],
        final_estimator=LogisticRegression(
            max_iter=req.max_iter_final, multi_class="multinomial"),
        stack_method="predict_proba",
        passthrough=False
    )

    new_stack.fit(X_train, y_train)
    y_pred = new_stack.predict(X_test)

    model_accuracy = accuracy_score(y_test, y_pred)
    model_recall = recall_score(y_test, y_pred, average="macro")
    model_precision = precision_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    if os.path.exists(f"{req.model_name}.joblib"):
        return

    joblib.dump(new_stack, f"{req.model_name}.joblib")
    with open(f"{req.model_name}_scores.json", "w") as f:
        score_data = {
            "accuracy_mean": cross["test_accuracy"].mean(),
            "precision_mean": cross["test_precision"].mean(),
            "recall_mean":  cross["test_recall"].mean(),
            "accuracy_std": cross["test_accuracy"].std(),
            "precision_std": cross["test_precision"].std(),
            "recall_std": cross["test_recall"].std(),
            #  "confusion_matrix": cm,
            "accuracy": model_accuracy,
            "precision": model_precision,
            "recall": model_recall,
            "score": score}
        json.dump(score_data, f)

    return score_data
