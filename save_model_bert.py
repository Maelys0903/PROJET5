import joblib
import mlflow
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from mlflow.models import infer_signature
from collections import Counter
import re

# 1. Chargement des données
df = pd.read_csv("stack_questions_api.csv")

# 2. Nettoyage
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

df["clean_title"] = df["title"].astype(str).apply(clean_text)
df["tags"] = df["tags"].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

# Garder les 50 tags les plus fréquents
all_tags = [tag for tags in df["tags"] for tag in tags]
top_tags = set([tag for tag, count in Counter(all_tags).most_common(50)])
df["tags"] = df["tags"].apply(lambda tags: [tag for tag in tags if tag in top_tags])
df = df[df["tags"].map(len) > 0]

# 3. Features et labels
X = df["clean_title"].tolist()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["tags"])

# 4. Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Embedding avec BERT
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_train = embedder.encode(X_train_raw, show_progress_bar=True)
X_test = embedder.encode(X_test_raw, show_progress_bar=True)

# 6. Modèle
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))
clf.fit(X_train, y_train)

# 7. Évaluation
y_pred = clf.predict(X_test)
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_samples = f1_score(y_test, y_pred, average='samples')
print("F1 micro:", f1_micro)
print("F1 samples:", f1_samples)

# 8. Wrapper pour MLflow
import numpy as np
from typing import Any

class BertWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, mlb):
        self.model = model
        self.mlb = mlb
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def predict(self, context: Any, model_input: pd.DataFrame) -> np.ndarray:
        texts = model_input["text"].tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        preds = self.model.predict(embeddings)
        return preds

# 9. Sauvegarde du modèle dans app/
local_model_path = Path("app/model_bert")
local_model_path.mkdir(parents=True, exist_ok=True)

input_example = pd.DataFrame({"text": X_test_raw[:5]})
signature = infer_signature(input_example)

mlflow.pyfunc.save_model(
    path=str(local_model_path),
    python_model=BertWrapper(clf, mlb),
    signature=signature,
    input_example=input_example
)

# 10. Sauvegarde du MultiLabelBinarizer
joblib.dump(mlb, "app/mlb.pkl")

print("Modèle BERT + LogisticRegression et mlb.pkl enregistrés dans app/")

# Terminal :
# Remove-Item -Recurse -Force .\app\model_bert
# python save_model_bert.py