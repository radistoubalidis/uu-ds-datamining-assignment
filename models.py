# ==========================================
# Gradient Boosting (like the other code, but GB)
# - Step A: pick best (vectorizer, n_features, ngram) by 5-fold CV on folds 1–4
# - Step B: GridSearch GB hyperparams for that setup, refit on folds 1–4
# - Step C: single final evaluation on fold5 + CSV logging + artifacts
# ==========================================
import os, re, string, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Paths ----------
DATA_ROOT = r"C:\Users\Thijm\Downloads\op_spam_v1.4 (1)"
OP_ROOT   = os.path.join(DATA_ROOT, "op_spam_v1.4")
CSV_PATH  = "dataset_df.csv"
OUT_CSV   = "gradBoost-accuracies.csv"

if not os.path.isdir(OP_ROOT):
    raise RuntimeError(f"'op_spam_v1.4' not found at: {OP_ROOT}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("dataset_df.csv must be next to this script/notebook.")

# ---------- Load + filter (negative only) ----------
df = pd.read_csv(CSV_PATH)
need = {"fold", "txt_path", "polarity"}
miss = need - set(df.columns)
if miss:
    raise ValueError(f"CSV missing columns: {miss}")

df = df[df["polarity"].astype(str).str.contains("negative", case=False)].copy()
if df.empty:
    raise RuntimeError("No rows after filtering to negative polarity.")

# Read text
texts, missing = [], []
for rel in df["txt_path"].astype(str):
    ap = os.path.join(DATA_ROOT, rel.replace("/", os.sep))
    if os.path.exists(ap):
        with open(ap, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    else:
        texts.append("")
        missing.append(ap)
if missing:
    raise RuntimeError(f"Could not read {len(missing)} files. First missing:\n{missing[0]}")

df["text"]  = texts
df["fold"]  = df["fold"].astype(str)
df["label"] = df["txt_path"].str.lower().map(lambda p: 0 if "deceptive" in p else 1)

# ---------- Preprocess helper (done inside vectorizer) ----------
STOP_WORDS = {"i","my","we","us","not","never","is","are","was","were",
              "could","would","might","should","hotel"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def clean_text(s: str) -> str:
    s = (s or "").translate(PUNCT_TABLE).lower()
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join([t for t in s.split() if t not in STOP_WORDS])

def make_vectorizer(kind: str, ngram: int, max_features: int):
    if kind == "count":
        return CountVectorizer(
            preprocessor=clean_text, strip_accents="unicode", lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            min_df=0.05, max_features=max_features, ngram_range=(1, ngram)
        )
    elif kind == "tf_noidf":
        return TfidfVectorizer(
            preprocessor=clean_text, strip_accents="unicode", lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            use_idf=False, norm="l2",
            min_df=0.05, max_features=max_features, ngram_range=(1, ngram)
        )
    else:
        raise ValueError("kind must be 'count' or 'tf_noidf'")

# ---------- Split: tune on folds 1–4, test on fold5 ----------
is_fold5 = df["fold"].str.lower().eq("fold5")
train_idx = df.index[~is_fold5].to_numpy()
test_idx  = df.index[ is_fold5].to_numpy()

X_train_text = df.loc[train_idx, "text"].tolist()
y_train      = df.loc[train_idx, "label"].to_numpy()
X_test_text  = df.loc[test_idx,  "text"].tolist()
y_test       = df.loc[test_idx,  "label"].to_numpy()

if len(X_test_text) == 0:
    raise RuntimeError("No rows in fold5 test split.")

# =====================================================
# Step A: Pick vectorizer setup by 5-fold CV (accuracy)
# =====================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

vec_options = [
    ("count",   1, 400),
    ("count",   2, 400),
    ("count",   1, 800),
    ("count",   2, 800),
    ("tf_noidf",1, 400),
    ("tf_noidf",2, 400),
]

# fixed, decent (non-optimal) GB config for this screening step
gb_screen = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.08, max_depth=3,
    subsample=0.9, min_samples_leaf=2, random_state=42
)

screen_rows = []
best_screen = {"cv_mean": -1, "setup": None, "vec": None}

for kind, ngram, maxF in vec_options:
    vec = make_vectorizer(kind, ngram, maxF)
    Xtr = vec.fit_transform(X_train_text)

    cv_scores = cross_val_score(gb_screen, Xtr, y_train, cv=skf, scoring="accuracy")
    cv_mean = float(np.mean(cv_scores))

    screen_rows.append({
        "vectorizer": kind,
        "ngram": ngram,
        "num_features": maxF,
        "cv_fold_1": float(cv_scores[0]),
        "cv_fold_2": float(cv_scores[1]),
        "cv_fold_3": float(cv_scores[2]),
        "cv_fold_4": float(cv_scores[3]),
        "cv_fold_5": float(cv_scores[4]),
        "cv_mean_acc": cv_mean,
    })

    if cv_mean > best_screen["cv_mean"]:
        best_screen = {"cv_mean": cv_mean, "setup": (kind, ngram, maxF), "vec": vec}

screen_df = pd.DataFrame(screen_rows).sort_values("cv_mean_acc", ascending=False)
print("\n[GB] Screening (vectorizer × ngrams × features) — top 3 by CV accuracy:")
print(screen_df.head(3).to_string(index=False))

# =====================================================
# Step B: GridSearch GB hyperparams for best vectorizer
# =====================================================
kind, ngram, maxF = best_screen["setup"]
vec_best = best_screen["vec"]

Xtr = vec_best.fit_transform(X_train_text)
Xte = vec_best.transform(X_test_text)

gb = GradientBoostingClassifier(random_state=42)
param_grid = {
    "n_estimators":     [100, 150, 200, 250],
    "learning_rate":    [0.10, 0.08, 0.06, 0.05],
    "max_depth":        [3, 4],
    "subsample":        [1.0, 0.9, 0.85],
    "min_samples_leaf": [1, 2, 3],
}
grid = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"},
    refit="accuracy",
    cv=skf,
    n_jobs=-1,
    verbose=0,
    return_train_score=False
)

t0 = time.time()
grid.fit(Xtr, y_train)
fit_seconds = round(time.time() - t0, 2)

best_gb   = grid.best_estimator_
best_params = grid.best_params_
best_cv_acc = float(grid.best_score_)

# Pull per-metric CV means for best row (for CSV)
cv_res = pd.DataFrame(grid.cv_results_)
best_row = cv_res.iloc[grid.best_index_]
avg_val_accuracy  = float(best_row["mean_test_accuracy"])
avg_val_precision = float(best_row["mean_test_precision"])
avg_val_recall    = float(best_row["mean_test_recall"])
avg_val_f1        = float(best_row["mean_test_f1"])

# =====================================================
# Step C: Final test on fold5 (single pass) + logging
# =====================================================
y_pred = best_gb.predict(Xte)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

with_bigrams = (ngram == 2)
exec_time    = datetime.now().isoformat(timespec="seconds")

print("\n=== Gradient Boosting FINAL (fold5) ===")
print(f"Vectorizer: {kind} | ngram={(1,ngram)} | max_features={maxF}")
print(f"Best CV (folds 1–4) accuracy: {best_cv_acc:.3f}")
print(f"fold5 -> acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
print("Best GB params:", best_params)

row = {
    "accuracy":           round(acc, 3),             # fold5 test accuracy (single split)
    "with_bigrams":       bool(with_bigrams),
    "execution_time":     exec_time,
    "num_features":       int(maxF),
    "k":                  skf.n_splits,              # CV folds used in tuning
    "avg_val_accuracy":   round(avg_val_accuracy, 3),
    "avg_val_precision":  round(avg_val_precision, 3),
    "avg_val_recall":     round(avg_val_recall, 3),
    "avg_val_f1":         round(avg_val_f1, 3),
    "fold":               "fold5",
    # traceability
    "vectorizer":         kind,
    "min_df":             0.05,
    "ngram_range":        str((1, ngram)),
    "n_estimators":       int(best_params["n_estimators"]),
    "learning_rate":      float(best_params["learning_rate"]),
    "max_depth":          int(best_params["max_depth"]),
    "subsample":          float(best_params["subsample"]),
    "min_samples_leaf":   int(best_params["min_samples_leaf"]),
    "fit_seconds":        fit_seconds,
}

pd.DataFrame([row]).to_csv(
    OUT_CSV,
    mode=("a" if os.path.exists(OUT_CSV) else "w"),
    header=(not os.path.exists(OUT_CSV)),
    index=False
)
print(f"\n✅ Appended 1 row to {os.path.abspath(OUT_CSV)}")

# ---------- Save confusion matrix & top-10 features ----------
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
pd.DataFrame(cm, index=["true_fake(0)","true_truthful(1)"],
                columns=["pred_fake(0)","pred_truthful(1)"]).to_csv("gradBoost-confusion.csv", index=True)
print("🧩 Saved confusion matrix -> gradBoost-confusion.csv")

feat_names = np.array(vec_best.get_feature_names_out())
imps = best_gb.feature_importances_
top_idx = np.argsort(imps)[::-1][:10]
with open("gradBoost-top10-features.txt", "w", encoding="utf-8") as f:
    f.write("Top 10 most important features (Gradient Boosting)\n")
    for w, s in zip(feat_names[top_idx], imps[top_idx]):
        f.write(f"{w}\t{s:.6f}\n")
print("📝 Saved top-10 features -> gradBoost-top10-features.txt")
