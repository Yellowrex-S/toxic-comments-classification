# Toxic Comment Classification for an Online Store

An NLP project from the Yandex Practicum Data Science program. The notebook is written in Russian; this README summarizes it in English.

## Business task

The online store «Викишоп» lets users edit product descriptions, wiki-style. It needs a tool that finds toxic comments and sends them to moderation.
Requirement: **F1 ≥ 0.75**.

## Data

`toxic_comments.csv`: 159,292 English comments labeled toxic / not toxic. About 10% are toxic, so the classes are imbalanced.

## What I did

1. **EDA.** No missing values or duplicates. Toxic comments are shorter (median 128 vs. 216 characters) and more often written in capitals. Differences confirmed with the Mann-Whitney test; correlations checked with φK.
2. **Text preprocessing.** Lowercasing, keeping only Latin letters, lemmatization with WordNet, English stop words removed.
3. **Modeling.** TF-IDF and the model in one `Pipeline`, so the vectorizer is fitted only on training folds (no leakage). `GridSearchCV` with F1; imbalance handled with `class_weight='balanced'` and a stratified split.

## Results

| Model | F1 (cross-validation) |
|---|---|
| **Logistic Regression (C = 5)** | **0.760** |
| LinearSVC (C = 0.5) | 0.756 |

- **F1 on the test set: 0.7585.** The requirement is met, and the result matches cross-validation, so there is no overfitting.
- Logistic regression on TF-IDF trains and predicts fast, needs no GPU and is interpretable by word weights. Next steps for higher quality: bigrams, threshold tuning or BERT-based models.

## Tech stack

Python · pandas · NumPy · NLTK · scikit-learn · SciPy · phik · Matplotlib · Seaborn

## How to run

```bash
pip install -r requirements.txt
jupyter notebook toxic_comments_classification.ipynb
```

---

**Кратко по-русски.** Классификация токсичных комментариев для интернет-магазина (учебный проект Яндекс Практикума): лемматизация, TF-IDF в пайплайне с моделью, учёт дисбаланса классов. Лучшая модель: логистическая регрессия, F1 на тесте 0,7585 при требовании не меньше 0,75.
