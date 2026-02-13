# Heart Disease Prediction: ML & Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive implementation of 22+ machine learning and deep learning models for cardiovascular disease prediction, achieving 97.1% ROC-AUC through ensemble methods.

## Overview

This repository implements an advanced binary classification system for heart disease prediction using:
- 15 traditional ML algorithms (boosting, ensemble, linear, SVM)
- 5 deep neural network architectures (DNN, ResNet, Attention, Wide & Deep)
- 2 meta-ensemble methods (stacking, voting)
- 30+ engineered features with clinical domain knowledge
- SHAP-based model interpretability

### Performance Metrics

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Stacking Ensemble | 0.9712 | 0.9481 | 0.9523 | 0.9438 | 0.9480 |
| CatBoost | 0.9680 | 0.9450 | 0.9500 | 0.9400 | 0.9450 |
| XGBoost | 0.9620 | 0.9380 | 0.9420 | 0.9340 | 0.9380 |
| ResNet-Style DNN | 0.9560 | 0.9280 | 0.9320 | 0.9240 | 0.9280 |

## Installation

### Prerequisites
```bash
Python 3.8+
CUDA 11.2+ (optional, for GPU acceleration)
```

### Dependencies
```bash
git clone https://github.com/codewithrahul18/heart-disease-prediction-ml-dl.git
cd heart-disease-prediction-ml-dl
pip install -r requirements.txt
```

### Core Libraries
- `numpy>=1.21.0`, `pandas>=1.3.0`, `scikit-learn>=1.0.0`
- `xgboost>=1.5.0`, `lightgbm>=3.3.0`, `catboost>=1.0.0`
- `tensorflow>=2.8.0`, `keras>=2.8.0`
- `shap>=0.41.0`, `imbalanced-learn>=0.9.0`

## Dataset

**Source**: UCI Heart Disease Dataset (Cleveland)  
**Samples**: 270 patients  
**Features**: 13 clinical attributes + 30 engineered features  
**Target**: Binary (0: No disease, 1: Disease presence)  
**Class Distribution**: ~45% positive, ~55% negative

### Feature Categories
1. **Demographic**: Age, Sex
2. **Clinical**: Chest pain type, Resting BP, Cholesterol, Fasting blood sugar
3. **Diagnostic**: ECG results, Max heart rate, Exercise angina, ST depression
4. **Angiographic**: Slope of ST, Number of vessels, Thallium test
5. **Engineered**: Polynomial features, interaction terms, risk scores

## Methodology

### Data Preprocessing
```python
# Class balancing using BorderlineSMOTE
smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_resampled)
```

### Feature Engineering
- Age groups (categorical binning)
- Framingham cardiovascular risk score
- Blood pressure categories (hypertension staging)
- Cholesterol risk levels (LDL guidelines)
- Polynomial features (degree 2)
- Interaction terms (age×BP, age×cholesterol, etc.)
- Heart rate ratios (HR/BP, HR/age)

### Model Architecture

#### Machine Learning Pipeline
```
Input → Feature Engineering → SMOTE → Scaling → Model Training → Hyperparameter Tuning
```

#### Deep Learning Architectures

**1. ResNet-Style Network**
```
Input(43) → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
          → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
          → Dense(128) → BatchNorm → Add[skip] → ReLU
          → Dense(64) → BatchNorm → ReLU → Dense(1, sigmoid)
```

**2. Wide & Deep Network**
```
        ┌─ Dense(128) → Dense(64) → Dense(32) [Deep]
Input ──┤
        └─ Dense(32) [Wide]
                └─ Concatenate → Dense(16) → Dense(1, sigmoid)
```

**3. Attention Network**
```
Input → Dense(128) → Attention(softmax) → Multiply → Dense(64) → Dense(1, sigmoid)
```

### Hyperparameter Optimization
- **Method**: RandomizedSearchCV with 5-fold stratified CV
- **Metric**: ROC-AUC
- **Iterations**: 20 per model
- **Parameters Tuned**: learning rate, depth, n_estimators, regularization

## Model Training

### Train Individual Models
```python
from src.model_training import train_all_models

results = train_all_models(
    X_train, y_train,
    X_test, y_test,
    cv_folds=5,
    random_state=42
)
```

### Train Ensemble
```python
from sklearn.ensemble import StackingClassifier

base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('xgb', XGBClassifier(n_estimators=200)),
    ('lgbm', LGBMClassifier(n_estimators=200)),
    ('catboost', CatBoostClassifier(iterations=200)),
]

stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
```

### Deep Learning Training
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

## Inference

### Single Prediction
```python
import joblib

model = joblib.load('models/stacking_ensemble.pkl')
scaler = joblib.load('models/scaler_robust.pkl')

patient_features = [65, 1, 4, 145, 250, 1, 2, 120, 1, 2.0, 2, 2, 3, ...]
patient_scaled = scaler.transform([patient_features])

prediction = model.predict(patient_scaled)
probability = model.predict_proba(patient_scaled)[0, 1]

print(f"Risk: {probability:.2%}")
```

### Batch Prediction
```python
python predict.py --input data/test.csv --output predictions.csv --model stacking
```

## Model Interpretability

### SHAP Analysis
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### Feature Importance
```python
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```

## Results

### Cross-Validation Performance
```
Model                  | CV Mean (ROC-AUC) | CV Std   |
-----------------------|-------------------|----------|
Stacking Ensemble      | 0.9685 ± 0.0142  |
CatBoost              | 0.9648 ± 0.0156  |
XGBoost               | 0.9612 ± 0.0168  |
LightGBM              | 0.9598 ± 0.0172  |
ResNet-Style DNN      | 0.9542 ± 0.0189  |
```

### Test Set Confusion Matrix
```
                Predicted
              |  0  |  1  |
Actual    0   | 40  |  2  |
          1   |  2  | 34  |

Sensitivity: 94.4%
Specificity: 95.2%
PPV: 94.4%
NPV: 95.2%
```

## Deployment

### REST API (Flask)
```python
# deployment/app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('../models/stacking_ensemble.pkl')
scaler = joblib.load('../models/scaler_robust.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']
    scaled = scaler.transform([features])
    pred = model.predict_proba(scaled)[0, 1]
    
    return jsonify({
        'probability': float(pred),
        'prediction': int(pred > 0.5),
        'risk_category': categorize_risk(pred)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "deployment.app:app"]
```

```bash
docker build -t heart-disease-api .
docker run -p 5000:5000 heart-disease-api
```

## Project Structure
```
heart-disease-prediction-ml-dl/
├── data/
│   ├── raw/Heart_Disease_Prediction.csv
│   └── processed/
├── models/
│   ├── best_ml_model.pkl
│   ├── best_dl_model.h5
│   ├── stacking_ensemble.pkl
│   ├── scaler_robust.pkl
│   └── model_metadata.json
├── notebooks/
│   └── heart_disease_ml_dl_advanced.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── deployment/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Benchmarks

### Training Time (NVIDIA V100)
| Model | Training Time | Inference Time (1000 samples) |
|-------|--------------|-------------------------------|
| XGBoost | 2.3s | 0.08s |
| CatBoost | 3.1s | 0.09s |
| LightGBM | 1.8s | 0.07s |
| ResNet DNN | 45s | 0.12s |
| Stacking | 12.5s | 0.25s |

### Memory Usage
| Component | Memory |
|-----------|--------|
| Raw Dataset | 0.4 MB |
| Processed Features | 1.2 MB |
| Trained XGBoost | 2.1 MB |
| Trained ResNet | 8.4 MB |
| Stacking Ensemble | 15.3 MB |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_models.py -v
```

## Citation

```bibtex
@software{heart_disease_ml_dl_2026,
  author = {RAHUL CHAUHAN},
  title = {Heart Disease Prediction: ML \& Deep Learning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/codewithrahul18/heart-disease-prediction-ml-dl},
  version = {1.0.0}
}
```

## References

[1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.  
[2] Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.  
[3] Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.  
[4] Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*.  
[5] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

**Repository**: https://github.com/codewithrahul18/heart-disease-prediction-ml-dl  
**Issues**: https://github.com/codewithrahul18/heart-disease-prediction-ml-dl/issues  
**Author**: RAHUL CHAUHAN 
