import re
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# Global constants for binary classification

CATEGORIES  = ["sport", "politics"]          # classes of interest
LABEL_MAP   = {0: "POLITICS", 1: "SPORT"}    # integer → human label
RANDOM_SEED = 42


# Load and explore BBC dataset, filter to sport/politics, encode labels

def load_and_explore(filepath: str) -> pd.DataFrame:
    """Load BBC dataset, filter to sport/politics categories, and encode labels."""
    df = pd.read_csv(filepath)

    required = {"category", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    print(f"Dataset: {len(df)} rows, {list(df.columns)}")
    print(f"Original distribution: {dict(df['category'].value_counts())}")

    before = len(df)
    df.dropna(subset=["category", "text"], inplace=True)
    if len(df) < before:
        print(f"\n  [INFO] Dropped {before - len(df)} rows with missing values.")

    df_filtered = df[df["category"].isin(CATEGORIES)].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    print(f"\n  Articles after filtering   : {len(df_filtered)}")
    print(f"  Category breakdown :")
    for cat, cnt in df_filtered["category"].value_counts().items():
        pct = cnt / len(df_filtered) * 100
        print(f"    {cat:<20} {cnt:>5} articles  ({pct:.1f} %)")

    df_filtered["label"] = df_filtered["category"].map({"sport": 1, "politics": 0})
    print(f"Filtered to {len(df_filtered)} articles ({dict(df_filtered['category'].value_counts())})")

    return df_filtered


# Curated set of ~200 common English stopwords (articles, prepositions, pronouns, etc.) to filter non-informative tokens
STOP_WORDS = {
    "a","about","above","after","again","against","all","also","although",
    "an","and","are","aren","as","at","back","be","because","been","before",
    "being","below","between","both","but","by","can","cannot","cent","come",
    "came","could","couldn","did","didn","do","does","don","done","down","dr",
    "during","each","eu","even","ever","every","few","first","five","for",
    "four","from","further","get","go","going","gone","got","had","has","hasn",
    "have","he","her","here","him","his","how","however","i","if","in","into",
    "is","isn","it","its","itself","just","know","last","like","ll","made",
    "make","many","may","me","might","more","most","mp","mr","mrs","ms","much",
    "my","myself","new","next","nine","no","nor","not","now","of","off","old",
    "on","once","only","or","other","our","out","over","own","per","pm","re",
    "said","same","say","says","see","seven","shall","she","should","shouldn",
    "since","six","so","some","still","such","s","t","take","taken","ten","than",
    "that","the","their","them","then","there","therefore","these","they",
    "think","this","those","though","three","through","thus","to","too","two",
    "uk","under","until","up","us","use","used","very","ve","was","wasn","we",
    "were","what","when","where","which","while","who","why","will","with","won",
    "wouldn","year","years","yet","you","your","d","m",
}


def _suffix_strip(token: str) -> str:
    """Apply rule-based suffix stripping for lightweight stemming without external libraries."""
    rules = [
        ("ational", "ate"),
        ("isation", "ise"),
        ("ization", "ize"),
        ("fulness", "ful"),
        ("ingness", "ing"),
        ("iveness", "ive"),
        ("nesses",  "ness"),
        ("ations",  "ate"),
        ("ments",   "ment"),
        ("ation",   "ate"),
        ("ising",   "ise"),
        ("izing",   "ize"),
        ("iness",   "ine"),
        ("ness",    ""),
        ("ment",    ""),
        ("tion",    ""),
        ("ing",     ""),
        ("ised",    "ise"),
        ("ized",    "ize"),
        ("ful",     ""),
        ("ive",     ""),
        ("ous",     ""),
        ("ers",     ""),
        ("ies",     "y"),
        ("est",     ""),
        ("ed",      ""),
        ("er",      ""),
        ("ly",      ""),
        ("es",      ""),
        ("s",       ""),
    ]
    for suffix, replacement in rules:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)] + replacement
    return token


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/non-alpha, tokenize, filter stopwords, and apply suffix stripping."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOP_WORDS]
    tokens = [_suffix_strip(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 3]
    return " ".join(tokens)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to all articles and create cleaned_text column."""
    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)
    all_tokens = " ".join(df["cleaned_text"]).split()
    vocab = set(all_tokens)
    print(f"Preprocessed: {len(all_tokens):,} tokens, {len(vocab):,} unique")
    return df


# Stratified 80/20 train-test split

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
):
    """Perform stratified 80/20 train-test split on preprocessed data."""
    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# Feature extraction: Bag-of-Words, TF-IDF, and Bigram TF-IDF

def get_vectorizers() -> dict:
    """Return three feature extractors: Bag-of-Words, TF-IDF, and Bigram TF-IDF."""
    return {
        "Bag_of_Words": CountVectorizer(
            max_features=8_000,
            min_df=2,
            ngram_range=(1, 1),
            strip_accents="unicode",
        ),
        "TF_IDF": TfidfVectorizer(
            max_features=8_000,
            min_df=2,
            sublinear_tf=True,
            ngram_range=(1, 1),
            strip_accents="unicode",
        ),
        "Bigrams_TF_IDF": TfidfVectorizer(
            max_features=15_000,
            min_df=2,
            sublinear_tf=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
        ),
    }


# Define three ML classifiers: Naive Bayes, Logistic Regression, SVM

def get_classifiers() -> dict:
    """Return three classifiers: Naive Bayes, Logistic Regression, and SVM."""
    return {
        "Naive_Bayes": MultinomialNB(
            alpha=0.1,
        ),
        "Logistic_Regression": LogisticRegression(
            C=1.0,
            max_iter=1_000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        ),
        "SVM": LinearSVC(
            C=1.0,
            max_iter=3_000,
            random_state=RANDOM_SEED,
        ),
    }


# Train and evaluate all 9 model-feature combinations with cross-validation

def _compute_metrics(y_true, y_pred, model_name: str, vec_name: str) -> dict:
    """Compute weighted-average precision, recall, F1 and accuracy."""
    return {
        "Model":     model_name,
        "Features":  vec_name,
        "Accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(
                         y_true, y_pred, average="weighted", zero_division=0
                     ) * 100, 2),
        "Recall":    round(recall_score(
                         y_true, y_pred, average="weighted", zero_division=0
                     ) * 100, 2),
        "F1_Score":  round(f1_score(
                         y_true, y_pred, average="weighted", zero_division=0
                     ) * 100, 2),
    }


def run_all_experiments(X_train, X_test, y_train, y_test):
    """Train all 9 model-feature combinations, evaluate with cross-validation, and return results."""
    print("\nTraining 9 model-feature combinations...")

    vectorizers = get_vectorizers()
    classifiers  = get_classifiers()
    results      = []
    pipelines    = {}

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for vec_name, vectorizer in vectorizers.items():
        for clf_name, classifier in classifiers.items():
            combo = f"{clf_name}  +  {vec_name}"

            pipe = Pipeline([
                ("vectorizer", clone(vectorizer)),
                ("classifier", clone(classifier)),
            ])

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            pipelines[combo] = (pipe, y_pred)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_splitter, scoring="f1_weighted")
            metrics = _compute_metrics(y_test, y_pred, clf_name, vec_name)
            metrics["CV_F1_Mean"] = round(cv_scores.mean() * 100, 2)
            metrics["CV_F1_Std"] = round(cv_scores.std() * 100, 2)
            results.append(metrics)

    results_df = (
        pd.DataFrame(results)
          .sort_values("F1_Score", ascending=False)
          .reset_index(drop=True)
    )
    return results_df, pipelines


# Show top discriminative features for linear models

def show_top_features(pipelines: dict, top_n: int = 15) -> None:
    """Print top discriminative words for each class from linear models."""
    print("\nTop Discriminative Features:")

    for combo, (pipe, _) in pipelines.items():
        clf = pipe.named_steps["classifier"]

        if not hasattr(clf, "coef_"):
            continue

        vec       = pipe.named_steps["vectorizer"]
        features  = vec.get_feature_names_out()
        coef      = clf.coef_[0]

        if len(features) != len(coef):
            min_len = min(len(features), len(coef))
            features = features[:min_len]
            coef = coef[:min_len]

        n_features = len(features)
        if n_features == 0:
            continue
        valid_top_n = min(top_n, n_features // 2)

        sorted_indices = np.argsort(coef)
        top_sport    = sorted_indices[-valid_top_n:][::-1]
        top_politics = sorted_indices[:valid_top_n]

        print(f"\n  {combo}")
        print(f"  {'Top SPORT words':<35}  {'Top POLITICS words'}")
        print(f"  {'─'*34}  {'─'*34}")
        for s_idx, p_idx in zip(top_sport, top_politics):
            if s_idx < len(features) and p_idx < len(features):
                print(f"  {features[s_idx]:<35}  {features[p_idx]}")
    print()


# Generate and save performance visualizations

def plot_results(results_df: pd.DataFrame, pipelines: dict, y_test) -> None:
    """Generate bar chart and confusion matrix grid visualizations."""
    import os
    print("\nGenerating visualizations...")

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    combo_labels = results_df["Model"] + "\n" + results_df["Features"]
    x     = np.arange(len(combo_labels))
    bar_w = 0.18

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    palette = ["#2E86AB", "#E84855", "#3BB273", "#F18F01"]

    for i, (metric, color) in enumerate(zip(metrics, palette)):
        offset = (i - 1.5) * bar_w
        bars = ax.bar(
            x + offset, results_df[metric], bar_w,
            label=metric, color=color, alpha=0.90,
            edgecolor="white", linewidth=0.8,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.18,
                f"{h:.1f}",
                ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, fontsize=9, color="#333333")
    ax.set_ylim(60, 108)
    ax.set_ylabel("Score  (%)", fontsize=12, labelpad=10)
    ax.set_title(
        "BBC Sport vs Politics Classifier\nAll Model × Feature Combinations",
        fontsize=14, fontweight="bold", pad=14, color="#222222",
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.40, color="#999999")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out1 = "model_comparison_bar_chart.png"
    plt.savefig(os.path.join(output_dir, out1), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out1}")

    n     = len(pipelines)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 5.2))
    fig.patch.set_facecolor("#F8F9FA")
    axes = axes.flatten()

    for idx, (combo, (_, y_pred)) in enumerate(pipelines.items()):
        ax  = axes[idx]
        cm  = confusion_matrix(y_test, y_pred)
        ax.set_facecolor("#F8F9FA")
        sns.heatmap(
            cm,
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=["Politics", "Sport"],
            yticklabels=["Politics", "Sport"],
            ax=ax,
            linewidths=0.8,
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "#155724" if i == j else "#721c24"
                ax.texts[i * cm.shape[1] + j].set_color(color)

        ax.set_title(combo, fontsize=8.5, fontweight="bold", pad=7, color="#222222")
        ax.set_xlabel("Predicted Label", fontsize=8, labelpad=6)
        ax.set_ylabel("True Label",      fontsize=8, labelpad=6)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Confusion Matrices — All Experiments",
        fontsize=14, fontweight="bold", y=1.01, color="#222222",
    )
    plt.tight_layout()

    out2 = "confusion_matrices.png"
    plt.savefig(
        os.path.join(output_dir, out2),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved → {out2}\n")


# Print ranked comparison table and highlight best combination

def print_summary_and_best(results_df: pd.DataFrame) -> None:
    """Print ranked comparison table and highlight best model combination."""
    print("\nResults (ranked by F1-Score):")
    display_cols = ["Model", "Features", "Accuracy", "Precision", "Recall", "F1_Score", "CV_F1_Mean", "CV_F1_Std"]
    print(results_df[display_cols].to_string(index=False))
    best = results_df.iloc[0]
    print(f"\nBest: {best['Model']} + {best['Features']} | F1: {best['F1_Score']}%")


# Predict class label for single raw article text

def predict_article(pipeline, text: str) -> str:
    """Preprocess and classify a single raw article using the trained pipeline."""
    cleaned = clean_text(text)
    pred    = pipeline.predict([cleaned])[0]
    return LABEL_MAP[pred]


# Main execution pipeline

if __name__ == "__main__":

    DATA_PATH = "bbc-text.csv"

    df = load_and_explore(DATA_PATH)

    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = split_data(df)

    results_df, pipelines = run_all_experiments(X_train, X_test, y_train, y_test)

    show_top_features(pipelines)

    plot_results(results_df, pipelines, y_test)

    print_summary_and_best(results_df)

    best_model = results_df.iloc[0]["Model"]
    best_features = results_df.iloc[0]["Features"]
    best_combo = f"{best_model} + {best_features}"
    best_pipeline, _ = pipelines[best_combo]

    print(f"\nPredictions using {best_combo}:")

    test_articles = [
        (
            "The striker scored a stunning last-minute penalty to win the "
            "championship final, sending fans into wild celebrations across "
            "the packed stadium."
        ),
        (
            "The prime minister announced sweeping new legislation as Parliament "
            "debated electoral reform and campaign finance donation cap limits."
        ),
        (
            "The team coach confirmed that three injured players will return "
            "for next week's crucial league derby match after successful rehab."
        ),
        (
            "Opposition MPs demanded a full parliamentary inquiry into government "
            "spending after the budget revealed a significant rise in the deficit."
        ),
        (
            "Federer retired from professional tennis after a glittering career "
            "spanning more than two decades and twenty Grand Slam singles titles."
        ),
        (
            "The chancellor outlined tax relief measures for businesses "
            "struggling with rising energy costs in the autumn statement today."
        ),
    ]

    for i, article in enumerate(test_articles, 1):
        label = predict_article(best_pipeline, article)
        print(f"  Article {i}: {article[:100]} ...")
        print(f"  Prediction: >>> {label} <<<")
        print()