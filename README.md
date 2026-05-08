🛒 Customer Shopping Behavior :
        An end-to-end machine learning pipeline that predicts **purchase amount** (regression) and **product category** (classification) from customer shopping behavior data — featuring full preprocessing, outlier removal, model evaluation, and cross-validation visualizations.

📌 Overview :
        This project tackles two real-world ML problems on a customer shopping dataset:
              - **Regression** — How much will a customer spend? (`Purchase Amount (USD)`)
              - **Classification** — What product category will they buy? (`Category`)
        The pipeline covers everything from raw data ingestion to model evaluation, making it a great reference for end-to-end ML workflows in Python.

✨ Features :
- 📊 Exploratory Data Analysis — Correlation heatmaps, null checks, shape inspection, and descriptive statistics.
- 🧹 Smart Preprocessing — Drops irrelevant columns, handles missing values, and removes outliers via Z-score filtering.
- 💰 Regression Models — Predicts purchase amount using Random Forest, XGBoost, and Linear Regression with log-transform target scaling.
- 🏷️ Classification Models — Predicts product category using 5 classifiers with full evaluation reports.
- 📈 Rich Visualizations — Heatmaps, per-model confusion matrices, and 5-fold cross-validation bar charts.
- 🔁 Reusable Structure — Modular sections that can be independently extended or swapped out.

🛠️ Tech Stack :

| Component            | Technology |

| 📊 Data Handling    | NumPy, Pandas |

| 📉 Visualization    | Matplotlib, Seaborn |

| 🤖 ML Models        | Scikit-learn, XGBoost |

| 📐 Statistics       | SciPy |

| 🔁 Validation       | StratifiedKFold, cross_val_score |


⚙️ How It Works :

🧹 Step 1 — Data Loading & EDA
- Loads `Data.csv` and prints shape, dtypes, null counts, and summary stats.
- Generates a **correlation heatmap** for all numeric features to understand relationships before modeling.

🔧 Step 2 — Preprocessing
- Drops non-informative columns: `S.No`, `Customer ID`, `Color`.
- Removes rows with missing values.
- Applies **Z-score outlier removal** (threshold = 3σ) across all numeric columns.

💰 Step 3 — Regression: Predicting `Purchase Amount (USD)`
- One-hot encodes all categorical features via `pd.get_dummies`.
- Applies `log1p` transform on the target variable to reduce skewness.
- Trains three models; predictions are inverse-transformed (`expm1`) before metric computation.


🏷️ Step 4 — Classification: Predicting `Category`
- Label-encodes the target and all remaining object columns.
- Uses a **70/30 sequential split** for train/test.
- Evaluates all five models with accuracy, full classification report, and a confusion matrix heatmap.

| Model                  | Notes |

| Decision Tree         | Interpretable baseline |
| Random Forest         | Ensemble, handles noise well |
| Perceptron            | Fast linear classifier |
| Logistic Regression   | Probabilistic linear model |
| MLP Classifier        | Neural net with logistic activation |

**Evaluation:** Accuracy · Precision · Recall · F1-score · Confusion Matrix

🔁 Step 5 — Cross-Validation
- Runs **5-fold Stratified K-Fold** cross-validation on every classifier.
- Plots per-fold accuracy as a bar chart with a dashed mean line for easy comparison.

👩‍💻 Author :
Harinitha —  B.Tech CSBS Student
