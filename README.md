# Bagging_Pipeline
Implemented BaggingClassifier with GridSearchCV hyperparameter tuning on real-world CSV Iris dataset, achieving production-ready ML pipeline with preprocessing and model optimization.


Dataset
Source: CSV file (150 samples × 5 columns)

Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm

Target: Species ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica') → LabelEncoded

Train/Test Split: 80/20 (120/30 samples)
Pipeline Architecture:

CSV → pandas → Preprocessing → Bagging Ensemble → Prediction
                ↓
        ├─ LabelEncoder (y)
        └─ Pipeline:
           ├─ std_scaler (StandardScaler)
           └─ bagging (BaggingClassifier)

Key Learnings
Pipeline prevents data leakage - scaling fits only on train folds

Step naming convention: stepname__parameter (e.g., bagging__n_estimators)

LabelEncoder for string targets outside pipeline (y preprocessing)

Bagging reduces variance - ensemble > single decision tree

GridSearchCV scales - 36 models in ~30 seconds
           

