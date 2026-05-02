# IntelSysProject: Intelligent Systems - Chronic Kidney Disease Prediction

## Overview

This project implements a comprehensive machine learning pipeline for predicting Chronic Kidney Disease (CKD) across three independent datasets. The pipeline includes data ingestion, cleaning, feature alignment, model experimentation with imbalance handling, feature importance analysis, and reduced feature set optimization.

## Project Structure

```
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── dataset_1.csv
│   │   ├── dataset_2.csv
│   │   └── dataset_3.csv
│   ├── cleaned/                      # Cleaned datasets from ingestion
│   │   ├── dataset_1/dataset_1_clean.csv
│   │   ├── dataset_2/dataset_2_clean.csv
│   │   └── dataset_3/dataset_3_clean.csv
│   └── aligned/                      # Feature-aligned datasets
│       ├── dataset_1_aligned.csv
│       ├── dataset_2_aligned.csv
│       └── dataset_3_aligned.csv
├── models/                           # Trained models and preprocessors
│   ├── numeric_features.joblib
│   ├── categorical_features.joblib
│   ├── dataset_1/
│   ├── dataset_2/
│   └── dataset_3/
└── src/                              # Jupyter notebooks
    ├── ingestion/
    │   ├── dataset_1.ipynb          # ← START HERE
    │   ├── dataset_2.ipynb
    │   └── dataset_3.ipynb
    ├── cross_dataset_experiment.ipynb
    ├── feature_importance_analysis.ipynb
    └── reduced_feature_analysis.ipynb
```

## Execution Workflow

The notebooks **must be executed in the following order** for the pipeline to work correctly:

### Stage 1: Data Ingestion & Cleaning (Run First)

**Purpose**: Load raw datasets, visualize data distributions, handle missing values, perform minor preprocessing, and prepare data for experimentation.

1. **[dataset_1.ipynb](src/ingestion/dataset_1.ipynb)**
   - Loads `data/raw/dataset_1.csv`
   - Analyzes missing values and data distributions
   - Performs basic cleaning and validation
   - Outputs: `data/cleaned/dataset_1/dataset_1_clean.csv`

2. **[dataset_2.ipynb](src/ingestion/dataset_2.ipynb)**
   - Loads `data/raw/dataset_2.csv`
   - Analyzes missing values and data distributions
   - Performs basic cleaning and validation
   - Outputs: `data/cleaned/dataset_2/dataset_2_clean.csv`

3. **[dataset_3.ipynb](src/ingestion/dataset_3.ipynb)**
   - Loads `data/raw/dataset_3.csv`
   - Analyzes missing values and data distributions
   - Performs basic cleaning and validation
   - Outputs: `data/cleaned/dataset_3/dataset_3_clean.csv`

**⚠️ Important**: All three ingestion notebooks must complete successfully before proceeding to Stage 2.

### Stage 2: Cross-Dataset Experimentation (Run Second)

**Purpose**: Prepare data for machine learning, train multiple models across all datasets with and without imbalance handling (class weighting and SMOTE), and evaluate performance.

**[cross_dataset_experiment.ipynb](src/cross_dataset_experiment.ipynb)**

- Loads cleaned datasets from Stage 1
- Aligns features across all three datasets (keeps only common columns)
- Converts categorical features to numeric encoding
- Trains three model types on each dataset:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost
- For each model, trains both:
  - **Base model** (standard training)
  - **Weighted model** (with class weighting and SMOTE for imbalance handling)
- Evaluates using: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Outputs: All trained models saved to `models/` directory

**Requirements**: Complete all ingestion notebooks first.

### Stage 3: Feature Importance Analysis (Run Third)

**Purpose**: Analyze which features are most important for CKD prediction using the Random Forest weighted models (best performers).

**[feature_importance_analysis.ipynb](src/feature_importance_analysis.ipynb)**

- Loads preprocessor and Random Forest weighted models from cross-dataset experiment
- Visualizes feature importance for each dataset
- Maps feature codes to readable clinical variable names:
  - `bp`: Blood pressure (mm/Hg)
  - `sg`: Specific gravity of urine
  - `al`: Albumin in urine
  - `su`: Sugar in urine
  - `bu`: Blood urea (mg/dl)
  - `sc`: Serum creatinine (mg/dl)
  - `sod`: Sodium level (mEq/L)
  - `pot`: Potassium level (mEq/L)
  - `hemo`: Hemoglobin level (gms)
  - `rbc`/`rc`: Red blood cell count (millions/cumm)
  - `wc`: White blood cell count (cells/cumm)
  - `htn`: Hypertension (yes/no)
- Compares importance patterns across datasets

**Requirements**: Complete cross-dataset experiment first.

**Key Findings**:

- **Datasets 1 & 2**: Hemoglobin, Serum Creatinine, and Specific Gravity are top predictors
- **Dataset 3**: Albumin and Sugar in urine are top predictors (different pattern)

### Stage 4: Reduced Feature Analysis (Run Last)

**Purpose**: Test model performance with incrementally larger feature subsets to identify optimal feature counts and understand feature contribution.

**[reduced_feature_analysis.ipynb](src/reduced_feature_analysis.ipynb)**

- Tests balanced and unbalanced Random Forest models
- Progressively adds features in order of importance (from Stage 3)
- Evaluates performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Generates elbow curves showing optimal feature counts
- Identifies performance plateaus

**Requirements**: Complete feature importance analysis first.

**Key Findings**:

- **Datasets 1 & 2**: Performance plateaus at ~5 features with minimal gains after
- **Dataset 3**: Limited signal overall; performance peaks early then declines
- **Imbalance Handling**: Balanced models provide more honest F1 scores on minority class

## Dependencies

All required libraries are installed within the notebooks using `%pip install`. The main dependencies are:

- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Model Persistence**: `joblib`
- **Notebook Utilities**: `IPython`

## Running the Pipeline

### Option 1: Sequential Manual Execution

1. Open and run all three ingestion notebooks in any order
2. Open and run `cross_dataset_experiment.ipynb`
3. Open and run `feature_importance_analysis.ipynb`
4. Open and run `reduced_feature_analysis.ipynb`

### Option 2: Quick Start (Recommended)

If models are already trained and saved in `models/`, you can skip ingestion and cross-dataset experiment:

- Jump directly to `feature_importance_analysis.ipynb`
- Then run `reduced_feature_analysis.ipynb`

## Models Saved

After completing the cross-dataset experiment, the following models are saved in `models/`:

For each dataset (1, 2, 3):

- `LogisticRegression_base.joblib` - Base LR model
- `LogisticRegression_weighted.joblib` - LR with class weighting + SMOTE
- `RandomForestClassifier_base.joblib` - Base RF model
- `RandomForestClassifier_weighted.joblib` - RF with class weighting + SMOTE ⭐ (used in analysis)
- `XGBClassifier_base.joblib` - Base XGB model
- `XGBClassifier_weighted.joblib` - XGB with class weighting + SMOTE
- `preprocessor.joblib` - Feature preprocessor (scaling, encoding)

Additional shared files:

- `numeric_features.joblib` - List of numeric feature names
- `categorical_features.joblib` - List of categorical feature names

## Key Insights

1. **Imbalance Matters**: Balanced models (with SMOTE) show significantly better F1 scores on the minority class compared to base models.

2. **Cross-Dataset Consistency**: Datasets 1 and 2 show consistent feature importance patterns, while Dataset 3 exhibits different predictive drivers.

3. **Feature Efficiency**: Strong predictive power can be achieved with only 5 key features across Datasets 1 and 2, suggesting high feature redundancy.

4. **Dataset Quality**: Dataset 3 shows weaker overall predictive performance, suggesting potential data quality or disease representation differences.

## Troubleshooting

- **FileNotFoundError**: Ensure you've completed all prior stages. Check that output files exist before proceeding to the next notebook.
- **Missing Models**: If running feature importance or reduced feature analysis, ensure cross-dataset experiment has completed and models are saved to `models/`.
- **Memory Issues**: Large datasets may require increased notebook kernel memory. Restart kernel if experiencing slowdowns.
