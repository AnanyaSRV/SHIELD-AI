# SHIELD-AI: A Multi-Agent Fraud Investigation Framework for Digital Payment Security

# 1\. Overview

# SHIELD-AI is a research prototype for machine-learning-based fraud risk assessment and investigation support. The currently tested workflow combines a tuned LightGBM model, SHAP-based explanations, evidence generation, configurable decision recommendations, and investigation reports.

# The project uses the IEEE-CIS Fraud Detection dataset. Graph investigation is not integrated into live transaction investigations. A separate exploratory graph experiment links transactions sharing card1; these are proxy relationships, not verified account identities.

# 2\. Objectives

# \- Assess transaction fraud risk using machine learning.

# \- Evaluate performance under class imbalance.

# \- Explain predictions using SHAP feature attributions.

# \- Generate evidence summaries and investigation reports.

# \- Recommend actions using configurable decision thresholds.

# \- Explore transaction relationships through a separate proxy-graph experiment.

# 3\. System Components

# Risk Assessment Agent

# Uses the tuned LightGBM model to estimate fraud probability and classify transactions using a threshold selected on validation data.

# Explainability and Evidence

# Uses SHAP feature attributions to describe features pushing the model output toward or away from fraud. SHAP values describe contributions to the model's raw output; they are not probability-point changes and do not establish causation.

# Decision Agent

# Recommends an action based on the configured decision policy:

# \- ALLOW: Below the configured monitoring threshold.

# \- MONITOR: Elevated risk requiring attention.

# \- BLOCK: At or above the configured blocking threshold.

# \- VERIFY: An available action where supported by the decision workflow.

# These are model-assisted recommendations, not confirmed fraud findings. The policy thresholds have not been validated for production use.

# Report Agent

# Generates a structured investigation report containing the model prediction, risk score, evidence, recommended action, and graph-investigation status.

# Fusion and Graph Investigation

# Fusion and graph investigation are part of the broader project architecture, but the full fusion workflow has not been verified as the active application path. The standalone graph experiment uses shared card1 values as exploratory proxy links. It does not confirm account ownership or fraud rings and is not connected to live decisions.

# 4\. Dataset

# The primary dataset is the IEEE-CIS Fraud Detection dataset.

# Expected files:

# \- data/train\_transaction.csv

# \- data/train\_identity.csv

# Reported transaction statistics:

# Property	Value

# Transaction records	590,540

# Legitimate transactions	569,877 (96.50%)

# Fraudulent transactions	20,663 (3.50%)

# The tuned model uses a 60% development, 20% validation, and 20% test split. Preprocessing is fitted on development data, and the classification threshold is selected on validation data. The dataset is highly imbalanced, so accuracy alone is insufficient.

# Obtain IEEE-CIS through its authorized Kaggle source. Dataset files may not be included in every project copy.

# 5\. Technology Stack

# \- Python 3.10

# \- LightGBM

# \- scikit-learn

# \- SHAP

# \- pandas and NumPy

# \- Streamlit

# \- NetworkX

# \- Matplotlib and SciPy

# \- XGBoost for relevant research experiments

# 6\. Environment Setup

# Run commands from the project root in Anaconda Prompt or Windows Command Prompt.

# Activate the environment:

# conda activate upi

# 

# 

# Install dependencies:

# python -m pip install -r requirements.txt

# 

# 

# The explainability workflow requires SHAP. If it is missing from requirements.txt, add:

# shap==0.46.0

# 

# 

# Then install it if necessary:

# python -m pip install shap==0.46.0

# 

# 

# Ensure the dataset and tuned model artifacts exist at their expected paths.

# 7\. Running the Application

# From the project root:

# streamlit run archive\\miniproject\\app.py

# 

# 

# The locally tested interface includes:

# 1\. Dashboard — Application overview.

# 2\. Fraud Detection — Upload a compatible transaction CSV and run an assessment.

# 3\. Fraud Attribution — Inspect the latest risk score and evidence.

# 4\. Case Report — View and download the investigation report.

# The model expects 392 input features. Uploaded data must match the expected schema and must not include isFraud. The file data\\shield\_ai\_test\_transactions.csv can be used for a small schema-compatible test upload if present.

# This is a research prototype, not a production payment-blocking system.

# 8\. Held-Out Model Results

# The tuned LightGBM model was evaluated on a held-out test set using a threshold selected on validation data.

# Metric	Held-out test result

# Accuracy	97.10%

# Precision	56.03%

# Recall	79.55%

# F1-score	65.75%

# ROC-AUC	96.26%

# Average precision	78.94%

# Confusion Matrix

# &#x09;Predicted legitimate	Predicted fraud

# Actual legitimate	111,396	2,580

# Actual fraud	845	3,287

# The test set contains 118,108 transactions, including 4,132 fraudulent transactions.

# The classification threshold is approximately 0.3713, selected on validation data to target at least 80% fraud recall. This is separate from the action-policy thresholds.

# Precision indicates that false positives remain a limitation. These are offline results, not live performance.

# 9\. Evaluation Artifacts

# Generated files are stored in reports\\final\_graphs\\.

# Artifact	Description

# roc\_curve.png	ROC curve

# precision\_recall\_curve.png	Precision–recall curve

# confusion\_matrix.png	Held-out test confusion matrix

# threshold\_tradeoff.png	Classification threshold trade-offs

# holdout\_policy\_tradeoff.png	Diagnostic policy comparison

# threshold\_metrics.csv	Threshold-level metrics

# graph\_summary.txt	Evaluation summary

# The holdout policy chart is diagnostic only. Do not use test-set results to select thresholds; select policies using validation data and assess them on an untouched test set.

# Other training, fusion, cross-dataset, ablation, and evaluation scripts are optional experiments. Check their inputs, dependencies, and methodology before running them.

# 10\. Decision Policy

# Action	Configured probability range

# ALLOW	Below 0.60

# MONITOR	0.60 to below 0.80

# BLOCK	0.80 or above

# These are research settings, not production-calibrated thresholds. They are separate from the model-classification threshold of approximately 0.3713.

# 11\. Known Limitations

# \- Live graph investigation is not integrated because uploaded transactions cannot currently be mapped reliably to graph nodes.

# \- The standalone graph uses shared card1 values as proxy links, not verified account identities.

# \- Fraud predictions can produce false positives and false negatives.

# \- SHAP explains model behavior, not causation.

# \- Decision thresholds need operational validation.

# \- Local tests do not establish production readiness.

# \- Performance on IEEE-CIS does not guarantee performance on other datasets or payment systems.

# \- Live transaction integration, operational monitoring, and real-world decision validation have not been established.

# 12\. Future Work

# \- Establish validated transaction and account identifiers for graph construction.

# \- Integrate and validate graph investigation before using graph findings in decisions.

# \- Evaluate fusion strategies with consistent, leakage-controlled splits.

# \- Calibrate probabilities and validate decision policies.

# \- Conduct cross-dataset and temporal evaluations.

# \- Improve false-positive handling, monitoring, audit logging, and deployment safeguards.

# 13\. Project Status

# The locally tested application supports tuned LightGBM risk assessment, SHAP explainability, evidence generation, decision recommendations, reports, and offline evaluation artifacts.

# The standalone proxy-graph experiment runs, but graph investigation is not integrated into live transaction investigations and does not establish confirmed fraud rings.

# Project: SHIELD-AI — A Multi-Agent Fraud Investigation Framework for Digital Payment Security

# Primary dataset: IEEE-CIS Fraud Detection

# Primary model: Tuned LightGBM

# Interface: Streamlit

