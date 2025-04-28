# Intrusion Detection on NSL-KDD Dataset using AutoEncoder

## Task description

This project explores the NSL-KDD dataset to build and evaluate an Autoencoder-based model for network intrusion detection. The Autoencoder, a neural network architecture, is trained on normal network connections to learn their patterns. Anomalies, such as network attacks, are detected by identifying instances with high reconstruction errors. Through this, we assess the effectiveness of a deep learning model in handling tabular data for anomaly detection. This notebook provides a step-by-step guide to data exploration, preprocessing, model implementation, and performance evaluation.

## Initial Results (Without Hyperparameter Tuning)

Actual Intrusion: 12833, Detected: 11210

Confusion Matrix:

[[ 8831   880]
 [ 2503 10330]]

Precision: 0.9215

Recall: 0.8050

F1-Score: 0.8593

ROC AUC: 0.9165