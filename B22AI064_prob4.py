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

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES  = ["sport", "politics"]          # classes of interest
LABEL_MAP   = {0: "POLITICS", 1: "SPORT"}    # integer → human label
RANDOM_SEED = 42


# =============================================================================
#  STEP 1 ── LOAD & EXPLORE DATA
# =============================================================================

def load_and_explore(filepath: str) -> pd.DataFrame:
    """
    Load bbc-text.csv, print dataset statistics,
    filter to sport and politics only, encode labels.

    Parameters
    ----------
    filepath : str
        Path to bbc-text.csv

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with added 'label' column (sport=1, politics=0)
    """
    df = pd.read_csv(filepath)

    # ── Validate expected columns ─────────────────────────────────────────────
    required = {"category", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    print("=" * 65)
    print("  STEP 1  ──  DATA LOADING & EXPLORATION")
    print("=" * 65)
    print(f"\n  Total rows in dataset      : {len(df)}")
    print(f"  Columns                    : {list(df.columns)}")
    print(f"\n  Full category distribution :")
    for cat, cnt in df["category"].value_counts().items():
        print(f"    {cat:<20} {cnt:>5} articles")

    # ── Drop rows with missing text or category ───────────────────────────────
    before = len(df)
    df.dropna(subset=["category", "text"], inplace=True)
    if len(df) < before:
        print(f"\n  [INFO] Dropped {before - len(df)} rows with missing values.")

    # ── Filter to target categories ───────────────────────────────────────────
    df_filtered = df[df["category"].isin(CATEGORIES)].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    print(f"\n  Articles after filtering   : {len(df_filtered)}")
    print(f"  Category breakdown :")
    for cat, cnt in df_filtered["category"].value_counts().items():
        pct = cnt / len(df_filtered) * 100
        print(f"    {cat:<20} {cnt:>5} articles  ({pct:.1f} %)")

    # ── Binary label encoding  (sport=1, politics=0) ──────────────────────────
    df_filtered["label"] = df_filtered["category"].map({"sport": 1, "politics": 0})

    # ── Print one representative sample from each class ───────────────────────
    print()
    for cat in CATEGORIES:
        sample = df_filtered.loc[df_filtered["category"] == cat, "text"].iloc[0]
        print(f"  ── Sample [{cat.upper()}] ──")
        print(f"  {sample[:260]} ...\n")

    # ── Basic text-length statistics ──────────────────────────────────────────
    df_filtered["text_length"] = df_filtered["text"].str.split().str.len()
    print("  Word-count statistics per category:")
    print(
        df_filtered.groupby("category")["text_length"]
        .agg(["mean", "min", "max"])
        .rename(columns={"mean": "Mean", "min": "Min", "max": "Max"})
        .round(1)
        .to_string()
    )
    print()

    return df_filtered


# =============================================================================
#  STEP 2 ── TEXT PREPROCESSING
# =============================================================================

# Custom English stopword list – no external library required
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
    """
    Apply simple rule-based suffix stripping (a lightweight alternative to
    stemming that requires no external libraries).
    Rules are applied in order from longest to shortest suffix.
    """
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
        # Only strip if the stem left behind is >= 3 characters
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)] + replacement
    return token


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline for one article string.

    Steps
    -----
    1.  Lowercase the text
    2.  Remove URLs  (http / www patterns)
    3.  Remove non-alphabetic characters (digits, punctuation, symbols)
    4.  Collapse consecutive whitespace to a single space
    5.  Tokenise by splitting on whitespace
    6.  Remove stopwords and tokens shorter than 3 characters
    7.  Apply lightweight suffix stripping (pseudo-stemmer)
    8.  Rejoin cleaned tokens into a single string

    Parameters
    ----------
    text : str
        Raw article text.

    Returns
    -------
    str
        Cleaned, normalised text ready for vectorisation.
    """
    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Step 3 – keep only a-z and whitespace
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4 – collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 5 – tokenise
    tokens = text.split()

    # Step 6 – remove stopwords and very short tokens
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOP_WORDS]

    # Step 7 – suffix stripping
    tokens = [_suffix_strip(t) for t in tokens]

    # Remove any tokens that became too short after stripping
    tokens = [t for t in tokens if len(t) >= 3]

    # Step 8 – rejoin
    return " ".join(tokens)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text to every article and store the result in
    the new column 'cleaned_text'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_and_explore().

    Returns
    -------
    pd.DataFrame
        DataFrame with the additional 'cleaned_text' column.
    """
    print("=" * 65)
    print("  STEP 2  ──  TEXT PREPROCESSING")
    print("=" * 65)

    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Sanity check – show one before / after pair
    idx = df.index[0]
    print("\n  Original text (first 220 chars) :")
    print(f"  {df.loc[idx, 'text'][:220]}")
    print("\n  Cleaned  text (first 220 chars) :")
    print(f"  {df.loc[idx, 'cleaned_text'][:220]}")

    # Vocabulary size after cleaning
    all_tokens = " ".join(df["cleaned_text"]).split()
    vocab = set(all_tokens)
    print(f"\n  Total tokens in corpus     : {len(all_tokens):,}")
    print(f"  Unique tokens (vocabulary) : {len(vocab):,}")
    print()

    return df


# =============================================================================
#  STEP 3 ── TRAIN / TEST SPLIT
# =============================================================================

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
):
    """
    Perform a stratified 80/20 train-test split.
    Stratification ensures both splits have the same class ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with 'cleaned_text' and 'label' columns.
    test_size : float
        Fraction of data reserved for testing (default 0.20).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("=" * 65)
    print("  STEP 3  ──  TRAIN / TEST SPLIT  (stratified 80 / 20)")
    print("=" * 65)
    print(f"\n  Total articles  : {len(df)}")
    print(f"  Training set    : {len(X_train)}  "
          f"(sport={sum(y_train==1)}, politics={sum(y_train==0)})")
    print(f"  Test set        : {len(X_test)}   "
          f"(sport={sum(y_test==1)}, politics={sum(y_test==0)})")
    print()

    return X_train, X_test, y_train, y_test


# =============================================================================
#  STEP 4 ── FEATURE REPRESENTATIONS
# =============================================================================

def get_vectorizers() -> dict:
    """
    Define three feature-extraction methods.

    1. Bag_of_Words
       Represents each document as a vector of raw word counts.
       Simple, interpretable, ignores word order.

    2. TF_IDF
       Weights each term by how frequent it is in the document (TF)
       scaled by how rare it is across all documents (IDF).
       Reduces the influence of very common words automatically.

    3. Bigrams_TF_IDF
       Extends TF-IDF to include consecutive word pairs (bigrams)
       e.g. "prime minister", "penalty kick".
       Captures contextual phrases missed by unigrams alone.

    Returns
    -------
    dict
        Mapping of name -> fitted-ready vectoriser instance.
    """
    return {
        "Bag_of_Words": CountVectorizer(
            max_features=8_000,
            min_df=2,                  # ignore terms appearing in < 2 documents
            ngram_range=(1, 1),
            strip_accents="unicode",
        ),
        "TF_IDF": TfidfVectorizer(
            max_features=8_000,
            min_df=2,
            sublinear_tf=True,         # use 1 + log(tf) to dampen high counts
            ngram_range=(1, 1),
            strip_accents="unicode",
        ),
        "Bigrams_TF_IDF": TfidfVectorizer(
            max_features=15_000,
            min_df=2,
            sublinear_tf=True,
            ngram_range=(1, 2),        # unigrams AND bigrams
            strip_accents="unicode",
        ),
    }


# =============================================================================
#  STEP 5 ── MACHINE LEARNING CLASSIFIERS
# =============================================================================

def get_classifiers() -> dict:
    """
    Define three ML classifiers.

    1. Naive_Bayes (MultinomialNB)
       Probabilistic model. Assumes feature independence.
       Very fast, works well with count / frequency features.
       alpha=0.1 gives Laplace smoothing to handle unseen words.

    2. Logistic_Regression
       Discriminative linear model. Learns a decision boundary.
       Strong baseline for text classification, highly interpretable.
       C=1.0 is default regularisation; lbfgs solver handles multi-class.

    3. SVM (LinearSVC)
       Support Vector Machine with a linear kernel.
       Maximises the margin between classes.
       Excellent performance on high-dimensional sparse text vectors.
       C=1.0 controls the trade-off between margin size and misclassification.

    Returns
    -------
    dict
        Mapping of name -> classifier instance.
    """
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


# =============================================================================
#  STEP 6 ── TRAIN, CROSS-VALIDATE & EVALUATE ALL 9 COMBINATIONS
# =============================================================================

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
    """
    Build a sklearn Pipeline for every (vectoriser × classifier) combination.
    Using a Pipeline is critical: it ensures the vectoriser is fit ONLY on the
    training data, preventing data leakage into the test set.

    For each combination:
      - Fit the pipeline on training data
      - Predict on the held-out test set
      - Print a detailed per-class classification report
      - Run 5-fold cross-validation on the training set for added confidence

    Parameters
    ----------
    X_train, X_test : pd.Series
        Cleaned article text for train and test splits.
    y_train, y_test : pd.Series
        Integer labels (0=politics, 1=sport).

    Returns
    -------
    results_df : pd.DataFrame
        All 9 combinations ranked by F1-Score.
    pipelines  : dict
        {combo_label: (trained_pipeline, test_predictions)}
    """
    print("=" * 65)
    print("  STEP 4-6  ──  FEATURE EXTRACTION + TRAINING + EVALUATION")
    print("=" * 65)

    vectorizers = get_vectorizers()
    classifiers  = get_classifiers()
    results      = []
    pipelines    = {}

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for vec_name, vectorizer in vectorizers.items():
        for clf_name, classifier in classifiers.items():

            combo = f"{clf_name}  +  {vec_name}"
            print(f"\n{'─'*65}")
            print(f"  Experiment : {combo}")
            print(f"{'─'*65}")

            # ── Build pipeline (prevents data leakage) ────────────────────────
            # clone() creates independent copies so pipelines don't share state
            pipe = Pipeline([
                ("vectorizer", clone(vectorizer)),
                ("classifier", clone(classifier)),
            ])

            # ── Train on full training split ──────────────────────────────────
            pipe.fit(X_train, y_train)

            # ── Predict on held-out test split ────────────────────────────────
            y_pred = pipe.predict(X_test)
            pipelines[combo] = (pipe, y_pred)

            # ── Classification report ─────────────────────────────────────────
            print(classification_report(
                y_test, y_pred,
                target_names=["Politics", "Sport"],
                digits=4,
            ))

            # ── 5-fold cross-validation on training data ──────────────────────
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                cv=cv_splitter,
                scoring="f1_weighted",
            )
            print(f"  5-Fold CV F1 : {cv_scores.mean()*100:.2f}% "
                  f"(+/- {cv_scores.std()*100:.2f}%)")

            # ── Collect scalar metrics ────────────────────────────────────────
            metrics = _compute_metrics(y_test, y_pred, clf_name, vec_name)
            metrics["CV_F1_Mean"] = round(cv_scores.mean() * 100, 2)
            metrics["CV_F1_Std"]  = round(cv_scores.std()  * 100, 2)
            results.append(metrics)

    results_df = (
        pd.DataFrame(results)
          .sort_values("F1_Score", ascending=False)
          .reset_index(drop=True)
    )
    return results_df, pipelines


# =============================================================================
#  STEP 7 ── TOP DISCRIMINATIVE FEATURES
# =============================================================================

def show_top_features(pipelines: dict, top_n: int = 15) -> None:
    """
    For each experiment that uses a Logistic Regression or linear SVM,
    print the most discriminative words for each class.
    Feature coefficients show which words push the model toward each label.

    Parameters
    ----------
    pipelines : dict
        Trained pipelines from run_all_experiments().
    top_n : int
        Number of top features to display per class (default 15).
    """
    print("=" * 65)
    print("  STEP 7a  ──  TOP DISCRIMINATIVE FEATURES")
    print("=" * 65)

    for combo, (pipe, _) in pipelines.items():
        clf = pipe.named_steps["classifier"]

        # Only models with linear coefficients support this
        if not hasattr(clf, "coef_"):
            continue

        vec       = pipe.named_steps["vectorizer"]
        features  = vec.get_feature_names_out()
        coef      = clf.coef_[0]        # 1D array for binary classification

        # Check that dimensions match
        if len(features) != len(coef):
            # Align to the smaller dimension
            min_len = min(len(features), len(coef))
            features = features[:min_len]
            coef = coef[:min_len]

        # Ensure indices are within valid bounds
        n_features = len(features)
        if n_features == 0:
            continue
        valid_top_n = min(top_n, n_features // 2)

        # Get sorted indices, then take top and bottom
        sorted_indices = np.argsort(coef)
        top_sport    = sorted_indices[-valid_top_n:][::-1]      # highest coefficients
        top_politics = sorted_indices[:valid_top_n]             # lowest coefficients

        print(f"\n  {combo}")
        print(f"  {'Top SPORT words':<35}  {'Top POLITICS words'}")
        print(f"  {'─'*34}  {'─'*34}")
        for s_idx, p_idx in zip(top_sport, top_politics):
            if s_idx < len(features) and p_idx < len(features):
                print(f"  {features[s_idx]:<35}  {features[p_idx]}")
    print()


# =============================================================================
#  STEP 8 ── VISUALISATIONS
# =============================================================================

def plot_results(results_df: pd.DataFrame, pipelines: dict, y_test) -> None:
    """
    Generate and save two publication-quality figures.

    Figure 1  ── Grouped bar chart: Accuracy, Precision, Recall, F1
                 for all 9 model × feature combinations.

    Figure 2  ── Grid of confusion matrices for all 9 experiments.

    Parameters
    ----------
    results_df : pd.DataFrame
        Metrics table from run_all_experiments().
    pipelines  : dict
        Trained pipelines keyed by combo label.
    y_test     : pd.Series
        True labels for the test split.
    """
    import os
    
    print("=" * 65)
    print("  STEP 8  ──  VISUALISATIONS")
    print("=" * 65)

    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ── Figure 1 : Grouped Bar Chart ──────────────────────────────────────────
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

    # ── Figure 2 : Confusion Matrices ─────────────────────────────────────────
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
        # Colour-code TP cells green, FP/FN cells red
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "#155724" if i == j else "#721c24"
                ax.texts[i * cm.shape[1] + j].set_color(color)

        ax.set_title(combo, fontsize=8.5, fontweight="bold", pad=7, color="#222222")
        ax.set_xlabel("Predicted Label", fontsize=8, labelpad=6)
        ax.set_ylabel("True Label",      fontsize=8, labelpad=6)

    # Hide unused subplot slots
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


# =============================================================================
#  STEP 9 ── SUMMARY TABLE & BEST MODEL
# =============================================================================

def print_summary_and_best(results_df: pd.DataFrame) -> None:
    """
    Print the full ranked comparison table and highlight the best combination.

    Parameters
    ----------
    results_df : pd.DataFrame
        Sorted metrics table from run_all_experiments().
    """
    print("=" * 65)
    print("  STEP 9  ──  FINAL COMPARISON TABLE  (ranked by F1-Score)")
    print("=" * 65)
    print()

    # Display all columns
    display_cols = ["Model", "Features", "Accuracy", "Precision",
                    "Recall", "F1_Score", "CV_F1_Mean", "CV_F1_Std"]
    print(results_df[display_cols].to_string(index=False))
    print()

    best = results_df.iloc[0]
    print("=" * 65)
    print("  BEST COMBINATION  (highest weighted F1-Score on test set) :")
    print(f"    Model            : {best['Model']}")
    print(f"    Feature method   : {best['Features']}")
    print(f"    Accuracy         : {best['Accuracy']} %")
    print(f"    Precision        : {best['Precision']} %")
    print(f"    Recall           : {best['Recall']} %")
    print(f"    F1-Score         : {best['F1_Score']} %")
    print(f"    CV F1 (5-fold)   : {best['CV_F1_Mean']} % ± {best['CV_F1_Std']} %")
    print("=" * 65)
    print()


# =============================================================================
#  STEP 10 ── PREDICT ON NEW UNSEEN ARTICLES
# =============================================================================

def predict_article(pipeline, text: str) -> str:
    """
    Preprocess and classify a single raw article string using a
    trained pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted Pipeline (vectoriser + classifier).
    text : str
        Raw (uncleaned) article text.

    Returns
    -------
    str
        Predicted class label: 'SPORT' or 'POLITICS'.
    """
    cleaned = clean_text(text)
    pred    = pipeline.predict([cleaned])[0]
    return LABEL_MAP[pred]


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    DATA_PATH = "bbc-text.csv"

    # ── Step 1 : Load and explore ─────────────────────────────────────────────
    df = load_and_explore(DATA_PATH)

    # ── Step 2 : Preprocess text ──────────────────────────────────────────────
    df = preprocess_dataframe(df)

    # ── Step 3 : Train / test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df)

    # ── Steps 4-6 : Feature extraction + training + evaluation ───────────────
    results_df, pipelines = run_all_experiments(X_train, X_test, y_train, y_test)

    # ── Step 7a : Top features for interpretable models ───────────────────────
    show_top_features(pipelines)

    # ── Step 8 : Save visualisations ─────────────────────────────────────────
    plot_results(results_df, pipelines, y_test)

    # ── Step 9 : Print summary table ─────────────────────────────────────────
    print_summary_and_best(results_df)

    # ── Step 10 : Predict on new unseen articles using best model ─────────────
    best_model = results_df.iloc[0]["Model"]
    best_features = results_df.iloc[0]["Features"]
    best_combo = f"{best_model}  +  {best_features}"
    best_pipeline, _ = pipelines[best_combo]

    print("=" * 65)
    print("  STEP 10  ──  PREDICTIONS ON NEW UNSEEN ARTICLES")
    print("  (using best model:", best_combo, ")")
    print("=" * 65)
    print()

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