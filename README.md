# SMS Spam Detection System

An end-to-end SMS spam classification application built using Machine Learning and NLP.

The system classifies SMS messages as **Spam or Ham**, displays a calibrated spam-risk score, and allows users to block confirmed spam senders.

---

## Project Overview

The initial model used TF-IDF with Multinomial Naive Bayes.

Although the baseline model achieved high precision, its spam recall was relatively low, meaning several spam messages were incorrectly classified as Ham.

To improve spam detection, the project was upgraded using:

- Character-level TF-IDF
- Linear Support Vector Machine
- Stratified Cross-Validation
- GridSearchCV
- Decision Threshold Optimization
- Probability Calibration
- Error Analysis

The final model was deployed using Streamlit with a simple sender-blocking workflow.

---

## Model Performance

| Metric      | Baseline Model | Optimized Model |
| ----------- | -------------: | --------------: |
| Accuracy    |         95.45% |          99.32% |
| Precision   |        100.00% |          98.44% |
| Recall      |         64.12% |          96.18% |
| F1-Score    |         78.14% |          97.30% |
| Missed Spam |             47 |               5 |

---

## Machine Learning Pipeline

```text
Raw SMS
   ↓
Character-level TF-IDF
   ↓
Tuned Linear SVM
   ↓
Optimized Decision Threshold
   ↓
Spam / Ham Prediction
```
