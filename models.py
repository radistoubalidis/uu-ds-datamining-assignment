# gb_only_assignment_paths_same.py
# Gradient Boosting only — with SAME PATHS as before.
# - Uses dataset_df.csv that lists (fold, txt_path, polarity)
# - Filters negative polarity
# - Tune on folds 1–4 (inner CV), test once on fold 5
# - Runs two reps: unigrams and unigrams+bigrams
# Outputs:
#   - gradBoost-accuracies.csv (2 rows; appended if exists)
#   - gb_predictions_fold5.csv (paired predictions)
#   - cm_gradboost_uni.png / cm_gradboost_unibi.png
#   - gb_top10_features_uni.csv / gb_top10_features_unibi.csv

import os, re, string, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=UserWarning)
RNG = 42

# -----------------------------
# PATHS (same structure as your previous script)
# -----------------------------
DATA_ROOT = r"C:\Users\Thijm\Downloads\op_spam_v1.4 (1)"
OP_ROOT   = os.path.join(DATA_ROOT, "op_spam_v1.4")
CSV_PATH  = "dataset_df.csv"

if not os.path.isdir(OP_ROOT):
    raise RuntimeError(f"'op_spam_v1.4' not found at: {OP_ROOT}\nFix DATA_ROOT if needed.")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("dataset_df.csv must be next to this script/notebook.")

# -----------------------------
# Text cleaning (match your earlier style)
# -----------------------------
STOP_WORDS = {
    "i","my","we","us","not","never","is","are","was","were",
    "could","would","might","should","hotel"
}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def clean_text(s: str) -> str:
    s = (s or "").translate(PUNCT_TABLE)       # remove punctuation
    s = s.lower()                               # lowercase
    s = re.sub(r"\d+", "", s)                   # remove digits
    s = re.sub(r"\s+", " ", s).strip()          # squash spaces
    toks = [t for t in s.split() if t not in STOP_WORDS]
    return " ".join(toks)

# -----------------------------
# Load CSV + read texts (relative paths under DATA_ROOT)
# -----------------------------
df = pd.read_csv(CSV_PATH)

need = {"fold", "txt_path", "polarity"}
miss = need - set(df.columns)
if miss:
    raise ValueError(f"CSV missing columns: {miss}")

# Focus on negative polarity only (as in your setup)
df = df[df["polarity"].astype(str).str.contains("negative", case=False)].copy()
if df.empty:
    raise RuntimeError("No rows after filtering to negative polarity.")

# Read text files
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

# -----------------------------
# Split: tune on folds 1–4, test once on fold 5
# -----------------------------
is_fold5   = df["fold"].str.lower().eq("fold5")
X_train_tx = df.loc[~is_fold5, "text"].tolist()
y_train    = df.loc[~is_fold5, "label"].to_numpy()
X_test_tx  = df.loc[ is_fold5, "text"].tolist()
y_test     = df.loc[ is_fold5, "label"].to_numpy()
test_keys  = df.loc[ is_fold5, "txt_path"].tolist()

if len(X_test_tx) == 0:
    raise RuntimeError("No rows in fold5 test split. Check 'fold' values in the CSV.")

# -----------------------------
# Small GB grid (simple & fast)
# -----------------------------
GB_GRID = [
    dict(n_estimators=100, learning_rate=0.10, max_depth=3, subsample=1.0,  min_samples_leaf=1),
    dict(n_estimators=200, learning_rate=0.08, max_depth=3, subsample=0.9,  min_samples_leaf=2),
    dict(n_estimators=200, learning_rate=0.06, max_depth=4, subsample=0.9,  min_samples_leaf=2),
]

# Common vectorizer knobs for fair comparison
MIN_DF = 0.02
MAX_FEATURES = 2000

def eval_metrics(y_true, y_pred):
    return dict(
        accuracy = accuracy_score(y_true, y_pred),
        precision= precision_score(y_true, y_pred, zero_division=0),
        recall   = recall_score(y_true, y_pred, zero_division=0),
        f1       = f1_score(y_true, y_pred, zero_division=0),
    )

def plot_cm(y_true, y_pred, title, out_png):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def run_gb_for_ngram(ngram):
    """
    Tune GB on folds 1–4 via inner CV (select by mean F1),
    then train on all folds 1–4 and test once on fold 5.
    Returns predictions and logging info.
    """
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

    # 1) Inner-CV tuning
    best_cfg = None
    best_cv_f1 = -1.0

    for cfg in GB_GRID:
        f1s = []
        for tr_idx, va_idx in inner_cv.split(X_train_tx, y_train):
            vec = TfidfVectorizer(
                preprocessor=clean_text,
                lowercase=False,
                token_pattern=r"(?u)\b\w+\b",
                strip_accents="unicode",
                ngram_range=ngram,
                min_df=MIN_DF,
                max_features=MAX_FEATURES
            )
            Xtr_sp = vec.fit_transform([X_train_tx[i] for i in tr_idx])
            Xva_sp = vec.transform([X_train_tx[i] for i in va_idx])

            # GradientBoosting requires dense arrays
            Xtr = Xtr_sp.toarray()
            Xva = Xva_sp.toarray()

            clf = GradientBoostingClassifier(random_state=RNG, **cfg)
            clf.fit(Xtr, y_train[tr_idx])
            pred = clf.predict(Xva)
            f1s.append(f1_score(y_train[va_idx], pred, zero_division=0))

        mean_f1 = float(np.mean(f1s))
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_cfg = cfg

    # 2) Train best on all folds 1–4 and evaluate on fold 5
    vec = TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        strip_accents="unicode",
        ngram_range=ngram,
        min_df=MIN_DF,
        max_features=MAX_FEATURES
    )
    Xtr_sp = vec.fit_transform(X_train_tx)
    Xte_sp = vec.transform(X_test_tx)
    Xtr = Xtr_sp.toarray()
    Xte = Xte_sp.toarray()

    clf = GradientBoostingClassifier(random_state=RNG, **best_cfg)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    metrics = eval_metrics(y_test, pred)

    # 3) Confusion matrix + Top-10 features
    tag = "unibi" if ngram == (1,2) else "uni"
    plot_cm(y_test, pred,
            f"Confusion Matrix — Gradient Boosting ({'uni+bi' if tag=='unibi' else 'uni'})",
            f"cm_gradboost_{tag}.png")

    feat_names = vec.get_feature_names_out()
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    pd.DataFrame({
        "feature": feat_names[top_idx],
        "importance": importances[top_idx]
    }).to_csv(f"gb_top10_features_{tag}.csv", index=False)

    return {
        "pred": pred,
        "metrics": metrics,
        "best_cfg": best_cfg,
        "cv_mean_f1": best_cv_f1,
        "with_bigrams": bool(ngram == (1,2))
    }

# -----------------------------
# Run two GB models: (1,1) and (1,2)
# -----------------------------
results_rows = []
preds_wide = pd.DataFrame({"txt_path": test_keys, "y_true": y_test})

for ngram in [(1,1), (1,2)]:
    out = run_gb_for_ngram(ngram)

    # Row for main log (append if file exists)
    results_rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "GradientBoosting",
        "with_bigrams": out["with_bigrams"],
        "min_df": MIN_DF,
        "max_features": MAX_FEATURES,
        "cv_mean_f1": round(float(out["cv_mean_f1"]), 3),
        "test_accuracy": round(float(out["metrics"]["accuracy"]), 3),
        "test_precision": round(float(out["metrics"]["precision"]), 3),
        "test_recall": round(float(out["metrics"]["recall"]), 3),
        "test_f1": round(float(out["metrics"]["f1"]), 3),
        "best_params": str(out["best_cfg"]),
        "fold": "fold5"
    })

    # Paired predictions for McNemar
    col = f"pred_gb_{'unibi' if out['with_bigrams'] else 'uni'}"
    preds_wide[col] = out["pred"]

# -----------------------------
# Save outputs
# -----------------------------
# Append-friendly accuracies file
main_csv = "gradBoost-accuracies.csv"
df_rows = pd.DataFrame(results_rows)
df_rows.to_csv(main_csv,
               mode=("a" if os.path.exists(main_csv) else "w"),
               header=(not os.path.exists(main_csv)),
               index=False)

# Paired predictions
preds_wide.to_csv("gb_predictions_fold5.csv", index=False)

print("\n✅ Done (Gradient Boosting only).")
print(f" - {main_csv} (two rows appended: uni & uni+bi)")
print(" - gb_predictions_fold5.csv (paired predictions for McNemar)")
print(" - cm_gradboost_uni.png / cm_gradboost_unibi.png")
print(" - gb_top10_features_uni.csv / gb_top10_features_unibi.csv")
