# BBC Sport vs Politics Classifier

A text classifier that categorises BBC news articles as **Sport** or **Politics** using classical machine learning.

## Dataset

[BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html) — 2,225 articles across 5 categories. Only Sport (511) and Politics (417) are used, giving **928 articles**.

## Approach

**3 Feature Representations × 3 Classifiers = 9 Experiments**

| Features | Classifiers |
|---|---|
| Bag of Words | Multinomial Naive Bayes |
| TF-IDF | Logistic Regression |
| Bigrams TF-IDF | Linear SVM |

## Results

| Model | Features | Accuracy | F1-Score | CV F1 (5-fold) |
|---|---|---|---|---|
| Naive Bayes | Bag of Words | **100.00%** | **100.00%** | 99.73% |
| SVM | TF-IDF | **100.00%** | **100.00%** | 99.73% |
| SVM | Bag of Words | 99.46% | 99.46% | 99.46% |
| LR | TF-IDF | 99.46% | 99.46% | 99.60% |

All 9 combinations achieve **>98.9% accuracy**.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

```bash
python3 B22AI064_prob4.py
```

Outputs:
- Dataset statistics and preprocessing summary
- Classification reports for all 9 experiments
- Top discriminative features per class
- Comparison table ranked by F1-Score
- Predictions on new unseen articles
- Plots saved to `outputs/`

## Project Structure

```
├── B22AI064_prob4.py          # Main classifier script
├── bbc-text.csv               # Dataset
├── report.tex                 # LaTeX report
├── test_bbc_classifier.py     # Unit tests
├── test_prediction.py         # Prediction tests
└── outputs/
    ├── model_comparison_bar_chart.png
    └── confusion_matrices.png
```

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn
