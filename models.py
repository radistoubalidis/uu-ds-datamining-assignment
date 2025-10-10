# ==========================================
# Gradient Boosting — RandomizedSearchCV + "Top-10 Memory" (fast)
# - Tune on folds 1–4 (5-fold CV), pick best by CV accuracy
# - Also evaluate best configs saved from prior runs (top-10 memory)
# - Refit on folds 1–4; test once on fold5
# - Append ONE summary row to gradBoost-accuracies.csv
# - Save/update top-10 history across runs to gradBoost-top10-history.csv
# - Save confusion matrix + top-10 features
# ==========================================
import os, re, string, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Paths (EDIT DATA_ROOT if needed) ----------
DATA_ROOT = r"C:\Users\Thijm\Downloads\op_spam_v1.4 (1)"
OP_ROOT   = os.path.join(DATA_ROOT, "op_spam_v1.4")
CSV_PATH  = "dataset_df.csv"

OUT_SUMMARY_CSV = "gradBoost-accuracies.csv"         # single-row per run summary (test fold5)
TOP10_HISTORY   = "gradBoost-top10-history.csv"      # cumulative memory of top-10 CV configs across runs

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

# Read text from disk
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

# ---------- Preprocess (done inside vectorizer) ----------
STOP_WORDS = {"i","my","we","us","not","never","is","are","was","were",
              "could","would","might","should","hotel"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def clean_text(s: str) -> str:
    s = (s or "").translate(PUNCT_TABLE).lower()
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join([t for t in s.split() if t not in STOP_WORDS])

def make_count():
    return CountVectorizer(
        preprocessor=clean_text, strip_accents="unicode", lowercase=False,
        token_pattern=r"(?u)\b\w+\b"
    )

def make_tf_noidf():
    # TF (no IDF) to mirror the other code’s tfidf_noidf variant
    return TfidfVectorizer(
        preprocessor=clean_text, strip_accents="unicode", lowercase=False,
        token_pattern=r"(?u)\b\w+\b", use_idf=False, norm="l2"
    )

# ---------- Split: tune on folds 1–4, test on fold5 ----------
is_fold5 = df["fold"].str.lower().eq("fold5")
X_train_text = df.loc[~is_fold5, "text"].tolist()
y_train      = df.loc[~is_fold5, "label"].to_numpy()
X_test_text  = df.loc[ is_fold5, "text"].tolist()
y_test       = df.loc[ is_fold5, "label"].to_numpy()
if len(X_test_text) == 0:
    raise RuntimeError("No rows in fold5 test split.")

# ---------- Pipeline skeleton ----------
BASE_SEED = 42
pipe = Pipeline(steps=[
    ("vec", make_count()),  # placeholder; actual choice also tuned
    ("gb", GradientBoostingClassifier(random_state=BASE_SEED))
])

# ---------- Search space ----------
param_distributions = {
    # Choose vectorizer kind
    "vec": [make_count(), make_tf_noidf()],
    # Vectorizer hyperparams
    "vec__ngram_range":   [(1,1), (1,2)],
    "vec__min_df":        [0.05, 0.02],
    "vec__max_features":  [400, 800],
    # GB hyperparams
    "gb__n_estimators":     [100, 150, 200, 250],
    "gb__learning_rate":    [0.10, 0.08, 0.06, 0.05],
    "gb__max_depth":        [3, 4],
    "gb__subsample":        [1.0, 0.9, 0.85],
    "gb__min_samples_leaf": [1, 2, 3],
}

# 5-fold CV on training-only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=BASE_SEED)

# Multi-metric eval, refit by accuracy
scoring = {
    "accuracy":  "accuracy",
    "precision": "precision",
    "recall":    "recall",
    "f1":        "f1",
}

# ---------- Utility: history I/O ----------
TOPK = 10

PARAM_COLS = [
    "vectorizer", "vec__ngram_range", "vec__min_df", "vec__max_features",
    "gb__n_estimators", "gb__learning_rate", "gb__max_depth",
    "gb__subsample", "gb__min_samples_leaf"
]

def _vec_kind_from_obj(obj) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    if isinstance(obj, TfidfVectorizer):
        return "tf_noidf"
    elif isinstance(obj, CountVectorizer):
        return "count"
    # Fallback to repr() sniffing if needed
    s = str(obj).lower()
    return "tf_noidf" if "tfidf" in s else "count"

def _make_vec(kind: str):
    return make_tf_noidf() if str(kind).lower() == "tf_noidf" else make_count()

def _row_from_params(params: dict, mean_cv_acc: float) -> dict:
    vec_kind = _vec_kind_from_obj(params["vec"]) if "vec" in params else (
        "tf_noidf" if isinstance(params.get("vectorizer"), TfidfVectorizer) else params.get("vectorizer","count")
    )
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mean_cv_accuracy": round(float(mean_cv_acc), 6),
        "vectorizer": vec_kind,
        "vec__ngram_range": tuple(params.get("vec__ngram_range", (1,1))),
        "vec__min_df": float(params.get("vec__min_df", 0.05)),
        "vec__max_features": int(params.get("vec__max_features", 400)),
        "gb__n_estimators": int(params.get("gb__n_estimators", 100)),
        "gb__learning_rate": float(params.get("gb__learning_rate", 0.1)),
        "gb__max_depth": int(params.get("gb__max_depth", 3)),
        "gb__subsample": float(params.get("gb__subsample", 1.0)),
        "gb__min_samples_leaf": int(params.get("gb__min_samples_leaf", 1)),
        "source": params.get("source", "random_search")
    }

def _save_top10_history(new_rows: pd.DataFrame):
    # Merge into existing history and keep global top-10 by mean_cv_accuracy
    if os.path.exists(TOP10_HISTORY):
        hist = pd.read_csv(TOP10_HISTORY)
    else:
        hist = pd.DataFrame(columns=["timestamp","mean_cv_accuracy","source"] + PARAM_COLS)

    # Ensure consistent dtypes
    for c in ["mean_cv_accuracy","vec__min_df","gb__learning_rate","gb__subsample"]:
        if c in new_rows.columns:
            new_rows[c] = new_rows[c].astype(float)
    for c in ["vec__max_features","gb__n_estimators","gb__max_depth","gb__min_samples_leaf"]:
        if c in new_rows.columns:
            new_rows[c] = new_rows[c].astype(int)

    merged = pd.concat([hist, new_rows], ignore_index=True)

    # Drop exact param duplicates keeping best score, then sort/keep top K
    dedup_keys = PARAM_COLS
    merged.sort_values(["mean_cv_accuracy","timestamp"], ascending=[False, False], inplace=True)
    merged = merged.drop_duplicates(subset=dedup_keys, keep="first")
    merged = merged.head(TOPK).reset_index(drop=True)

    merged.to_csv(TOP10_HISTORY, index=False)
    return merged

def _load_seed_configs():
    if not os.path.exists(TOP10_HISTORY):
        return []
    hist = pd.read_csv(TOP10_HISTORY)
    # keep only parameter columns (convert tuple-like text for ngram_range)
    seeds = []
    for _, r in hist.iterrows():
        ngr = r["vec__ngram_range"]
        if isinstance(ngr, str):
            # parse "(1, 2)" -> (1,2)
            ngr = tuple(int(x) for x in re.findall(r"-?\d+", ngr))  # (1,2) or (1, 1)
            if len(ngr) == 1: ngr = (ngr[0], ngr[0])
        seed = {
            "vectorizer":       r["vectorizer"],
            "vec__ngram_range": ngr,
            "vec__min_df":      float(r["vec__min_df"]),
            "vec__max_features":int(r["vec__max_features"]),
            "gb__n_estimators": int(r["gb__n_estimators"]),
            "gb__learning_rate":float(r["gb__learning_rate"]),
            "gb__max_depth":    int(r["gb__max_depth"]),
            "gb__subsample":    float(r["gb__subsample"]),
            "gb__min_samples_leaf": int(r["gb__min_samples_leaf"]),
        }
        seeds.append(seed)
    return seeds

# ---------- Stage A: Evaluate seeds from history (if any) ----------
seed_configs = _load_seed_configs()

seed_best = None
seed_best_cv = -np.inf
seed_best_scores = None

if seed_configs:
    print(f"🔁 Loaded {len(seed_configs)} prior top configs from {TOP10_HISTORY}")
    for i, cfg in enumerate(seed_configs, 1):
        vec = _make_vec(cfg["vectorizer"])
        pipe_seed = Pipeline([
            ("vec", vec),
            ("gb", GradientBoostingClassifier(random_state=BASE_SEED))
        ])
        pipe_seed.set_params(**{
            "vec__ngram_range": cfg["vec__ngram_range"],
            "vec__min_df": cfg["vec__min_df"],
            "vec__max_features": cfg["vec__max_features"],
            "gb__n_estimators": cfg["gb__n_estimators"],
            "gb__learning_rate": cfg["gb__learning_rate"],
            "gb__max_depth": cfg["gb__max_depth"],
            "gb__subsample": cfg["gb__subsample"],
            "gb__min_samples_leaf": cfg["gb__min_samples_leaf"],
        })
        scores = cross_val_score(pipe_seed, X_train_text, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_acc = float(scores.mean())
        print(f"  Seed {i:02d} ({cfg['vectorizer']}): CV acc={mean_acc:.4f}")

        if mean_acc > seed_best_cv:
            seed_best_cv = mean_acc
            seed_best = (pipe_seed, cfg)
            seed_best_scores = scores

# ---------- Stage B: Randomized search as usual ----------
N_ITER = 50  # 40–60 recommended
search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring=scoring,
    refit="accuracy",
    cv=cv,
    random_state=BASE_SEED,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)

t0 = time.time()
search.fit(X_train_text, y_train)
fit_seconds = round(time.time() - t0, 2)

rand_best_est   = search.best_estimator_
rand_best_params= search.best_params_
rand_best_cv    = float(search.best_score_)

print(f"\n🎲 Random search best CV accuracy: {rand_best_cv:.4f}")

# ---------- Decide winner between seeds and random search ----------
if seed_best is not None and seed_best_cv > rand_best_cv:
    winner_source = "seed_history"
    best_cv_acc   = seed_best_cv
    best_est      = seed_best[0]
    # Also record "params-like" dict in the same shape as RandomizedSearchCV.best_params_
    best_params   = {
        "vec": _make_vec(seed_best[1]["vectorizer"]),
        "vec__ngram_range": seed_best[1]["vec__ngram_range"],
        "vec__min_df": seed_best[1]["vec__min_df"],
        "vec__max_features": seed_best[1]["vec__max_features"],
        "gb__n_estimators": seed_best[1]["gb__n_estimators"],
        "gb__learning_rate": seed_best[1]["gb__learning_rate"],
        "gb__max_depth": seed_best[1]["gb__max_depth"],
        "gb__subsample": seed_best[1]["gb__subsample"],
        "gb__min_samples_leaf": seed_best[1]["gb__min_samples_leaf"],
    }
    # For CSV fields below, emulate cv means using random search best row structure where needed
    cv_res = pd.DataFrame(search.cv_results_)  # still available for saving top-10 from random stage
    avg_val_accuracy  = best_cv_acc
    avg_val_precision = np.nan
    avg_val_recall    = np.nan
    avg_val_f1        = np.nan
else:
    winner_source = "random_search"
    best_cv_acc   = rand_best_cv
    best_est      = rand_best_est
    best_params   = rand_best_params
    cv_res = pd.DataFrame(search.cv_results_)
    best_row = cv_res.iloc[search.best_index_]
    avg_val_accuracy  = float(best_row["mean_test_accuracy"])
    avg_val_precision = float(best_row["mean_test_precision"])
    avg_val_recall    = float(best_row["mean_test_recall"])
    avg_val_f1        = float(best_row["mean_test_f1"])

print(f"🏁 Winner: {winner_source} with CV acc={best_cv_acc:.4f}")

# ---------- Evaluate winner on fold5 ----------
y_pred = best_est.predict(X_test_text)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

print("\n=== Gradient Boosting: FINAL (fold5) ===")
print(f"Best CV accuracy (folds 1–4): {best_cv_acc:.3f}")
print(f"Test on fold5: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
print("Best params:")
for k, v in best_params.items():
    if k == "vec":
        print(f"  vectorizer: {'tf_noidf' if isinstance(v, TfidfVectorizer) else 'count'}")
    else:
        print(f"  {k}: {v}")

# ---------- Append one row (assignment-style) ----------
with_bigrams = (best_params["vec__ngram_range"] == (1,2))
num_features = int(best_params.get("vec__max_features", 0))

row = {
    "accuracy":           round(acc, 3),             # fold5 test accuracy
    "with_bigrams":       bool(with_bigrams),
    "execution_time":     datetime.now().isoformat(timespec="seconds"),
    "num_features":       num_features,
    "k":                  cv.n_splits,               # CV folds used in tuning
    "avg_val_accuracy":   round(float(avg_val_accuracy), 3) if not np.isnan(avg_val_accuracy) else None,
    "avg_val_precision":  round(float(avg_val_precision), 3) if not np.isnan(avg_val_precision) else None,
    "avg_val_recall":     round(float(avg_val_recall), 3) if not np.isnan(avg_val_recall) else None,
    "avg_val_f1":         round(float(avg_val_f1), 3) if not np.isnan(avg_val_f1) else None,
    "fold":               "fold5",
    # Params for traceability
    "vectorizer":         ("tf_noidf" if isinstance(best_est.named_steps["vec"], TfidfVectorizer) else "count"),
    "min_df":             float(best_params["vec__min_df"]),
    "ngram_range":        str(tuple(best_params["vec__ngram_range"])),
    "n_estimators":       int(best_params["gb__n_estimators"]),
    "learning_rate":      float(best_params["gb__learning_rate"]),
    "max_depth":          int(best_params["gb__max_depth"]),
    "subsample":          float(best_params["gb__subsample"]),
    "min_samples_leaf":   int(best_params["gb__min_samples_leaf"]),
    "fit_seconds":        fit_seconds,
    "picked_from":        winner_source
}

pd.DataFrame([row]).to_csv(
    OUT_SUMMARY_CSV,
    mode=("a" if os.path.exists(OUT_SUMMARY_CSV) else "w"),
    header=(not os.path.exists(OUT_SUMMARY_CSV)),
    index=False
)
print(f"\n✅ Appended 1 row to {os.path.abspath(OUT_SUMMARY_CSV)}")

# ---------- Update/save TOP-10 history across runs ----------
# 1) Take the random-search CV results' top-10 from this run
top10_rows = []
if "rank_test_accuracy" in cv_res.columns:
    cv_sorted = cv_res.sort_values("mean_test_accuracy", ascending=False)
    for _, r in cv_sorted.head(10).iterrows():
        params = {
            "vec": r["param_vec"],
            "vec__ngram_range": tuple(r["param_vec__ngram_range"]),
            "vec__min_df": float(r["param_vec__min_df"]),
            "vec__max_features": int(r["param_vec__max_features"]),
            "gb__n_estimators": int(r["param_gb__n_estimators"]),
            "gb__learning_rate": float(r["param_gb__learning_rate"]),
            "gb__max_depth": int(r["param_gb__max_depth"]),
            "gb__subsample": float(r["param_gb__subsample"]),
            "gb__min_samples_leaf": int(r["param_gb__min_samples_leaf"]),
            "source": "random_search"
        }
        top10_rows.append(_row_from_params(params, float(r["mean_test_accuracy"])))

# 2) Also include the seed-winner (if winner was from history and not already in cv_res)
if winner_source == "seed_history":
    seed_cfg = seed_best[1].copy()
    # Rebuild a params-like dict with a vec object for pretty printing and saving
    params_like = {"vec": _make_vec(seed_cfg["vectorizer"]), **seed_cfg, "source": "seed_history"}
    top10_rows.append(_row_from_params(params_like, best_cv_acc))

if top10_rows:
    top10_df = pd.DataFrame(top10_rows)
    hist_df = _save_top10_history(top10_df)
    print(f"💾 Updated top-10 history -> {os.path.abspath(TOP10_HISTORY)}")
    print(hist_df[ ['mean_cv_accuracy','vectorizer','vec__ngram_range',
                    'vec__min_df','vec__max_features',
                    'gb__n_estimators','gb__learning_rate','gb__max_depth',
                    'gb__subsample','gb__min_samples_leaf'] ])

# ---------- Save confusion matrix (fold5) ----------
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
cm_df = pd.DataFrame(cm, index=["true_fake(0)","true_truthful(1)"],
                        columns=["pred_fake(0)","pred_truthful(1)"])
cm_df.to_csv("gradBoost-confusion.csv", index=True)
print("🧩 Saved confusion matrix -> gradBoost-confusion.csv")

# ---------- Save top-10 features by importance ----------
vec = best_est.named_steps["vec"]
feat_names = np.array(vec.get_feature_names_out())
gb = best_est.named_steps["gb"]
imps = gb.feature_importances_
top_idx = np.argsort(imps)[::-1][:10]

with open("gradBoost-top10-features.txt", "w", encoding="utf-8") as f:
    f.write("Top 10 most important features (Gradient Boosting)\n")
    for w, s in zip(feat_names[top_idx], imps[top_idx]):
        f.write(f"{w}\t{s:.6f}\n")
print("📝 Saved top-10 features -> gradBoost-top10-features.txt")
