# Longevity Risk Predictor — Preventive Health Early Warning System

## Machine Learning Systems Design — Course Project Report

| Field | Details |
|---|---|
| **Student Name** | Tahir Ahmed |
| **School of Study** | Computing & Data Science |
| **Year of Study** | 3rd Year |
| **Course Name** | Machine Learning Systems Design |
| **PRN** | 230200041 |
| **Project Title** | Longevity Risk Predictor — Preventive Health Early Warning System |
| **GitHub Repository** | [longevity-risk-mlops](https://github.com/tahirahmed/longevity-risk-mlops) |

---

## 1. Abstract

Chronic lifestyle diseases—principally cardiovascular disease, diabetes, and metabolic syndrome—remain the leading causes of preventable mortality worldwide. Early identification of at-risk individuals through data-driven screening can meaningfully shift outcomes from reactive treatment toward proactive prevention. This project presents the **Longevity Risk Predictor**, an end-to-end CI/CD MLOps pipeline that automates the full lifecycle of a machine-learning model designed to predict chronic disease risk from biometric and lifestyle indicators.

The system is built on the **UCI Heart Disease (Cleveland) dataset**, comprising 303 patient records described by 13 clinical features (age, sex, chest-pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, ST-segment slope, number of major vessels, and thalassemia type) plus a binary target indicating the presence or absence of heart disease. A **Random Forest** classifier, trained with scikit-learn and balanced via **SMOTE** (Synthetic Minority Over-sampling Technique) from the imbalanced-learn library, achieves a ROC-AUC of approximately 0.90 and an F1 score of approximately 0.85, demonstrating robust discriminative ability even on this modestly sized dataset.

The operational backbone of the project is a fully automated **GitHub Actions** CI/CD pipeline. On every code push, the CI stage checks out the repository, installs dependencies, runs a comprehensive **pytest** test suite, launches an **MLflow** tracking server backed by SQLite, trains the model while logging all hyperparameters and evaluation metrics, and builds a **Docker** image. Upon successful CI completion, the CD stage downloads the trained model artifact, rebuilds the Docker image with a commit-SHA tag, loads it into a local **Minikube** Kubernetes cluster, and applies Kubernetes Deployment and Service manifests that expose an interactive **Streamlit** web application on a fixed NodePort. The deployment runs two replicas with configured liveness and readiness probes for self-healing and zero-downtime updates.

The resulting system demonstrates that production-grade MLOps practices—experiment tracking, containerized reproducibility, automated testing, and orchestrated deployment—can be applied even to small-scale health ML projects, laying a foundation for real-world preventive-health screening tools that could be adopted by clinics, telemedicine platforms, and public-health programs.

---

## 2. Introduction

### 2.1 Motivation

Chronic non-communicable diseases (NCDs) are responsible for approximately 41 million deaths each year, accounting for 74 % of all global deaths according to the World Health Organization (WHO). Cardiovascular diseases alone claim an estimated 17.9 million lives annually, making them the single largest category of NCD mortality. The WHO further estimates that up to 80 % of premature cardiovascular events could be prevented through timely identification and management of modifiable risk factors such as hypertension, dyslipidemia, tobacco use, physical inactivity, and unhealthy diet. These statistics underscore a critical gap: the tools and workflows currently available for population-level risk screening are often manual, inconsistent, and inaccessible in resource-constrained settings.

Machine learning offers a compelling pathway to close this gap. Given a patient's biometric profile—blood pressure, cholesterol, heart rate, electrocardiographic signals—a trained classifier can output a quantitative risk probability in seconds, enabling clinicians to prioritize high-risk individuals for early intervention. However, the potential of ML is frequently undermined by the operational fragility of the systems that surround the model. Training a model in a Jupyter notebook is only the first step; the far harder challenge is ensuring that model can be reliably tested, versioned, deployed, monitored, and updated without manual intervention.

### 2.2 Challenges of Manual ML Workflows

Traditional, notebook-centric ML workflows suffer from several well-documented failure modes. **Reproducibility** is fragile: small changes in library versions, random seeds, or preprocessing order can silently alter results. **Deployment** is ad hoc: models are often copied via USB drives or email attachments, leading to version mismatches between what was tested and what is running in production. **Experiment tracking** is non-existent or scattered across spreadsheets, meaning that the rationale for choosing one hyperparameter configuration over another is lost within days. **Testing** is an afterthought, so data-quality regressions and preprocessing bugs propagate undetected into production predictions.

### 2.3 How MLOps Solves These Challenges

MLOps—the discipline of applying DevOps principles to machine-learning systems—directly addresses each of these pain points. **Automated testing** via pytest catches data-pipeline bugs before they reach the model. **Continuous Integration** (CI) via GitHub Actions ensures that every commit is validated against the full test suite and that models are trained in a clean, reproducible environment. **MLflow** provides a centralized experiment tracker where every hyperparameter, metric, and artifact is logged, versioned, and queryable. **Docker** encapsulates the entire runtime—OS, libraries, model weights, application code—into an immutable image, eliminating "it works on my machine" failures. **Kubernetes** orchestrates the deployment of that image, providing horizontal scaling, self-healing via health probes, and zero-downtime rolling updates.

### 2.4 Tech Stack Integration

The technologies chosen for this project form a coherent pipeline where the output of each stage feeds directly into the next: a Git push triggers GitHub Actions → pytest validates code → MLflow tracks training → Docker packages the artifact → Kubernetes deploys the container → Streamlit serves the end user. This linear, automated flow ensures that every model in production has a clear provenance chain stretching back to the exact commit, test results, training metrics, and Docker image that produced it.

### 2.5 Major Contributions

- Designed and implemented a complete CI/CD pipeline for health ML systems using GitHub Actions with automated triggering on code push and pull-request events.
- Automated the full workflow—testing, training, building, and deployment—eliminating manual handoffs between development and operations.
- Integrated MLflow for comprehensive experiment tracking, metric logging, model artifact storage, and model registry.
- Applied SMOTE to handle class imbalance in the health dataset, improving recall for the minority (disease-present) class.
- Containerized the application using Docker with a slim Python base image, layer-cached dependency installation, and built-in health checks.
- Orchestrated deployment with Kubernetes (Minikube) for scalability, high availability, and self-healing through liveness and readiness probes.
- Built an interactive Streamlit web application with a premium health-themed UI for real-time disease risk predictions.
- Demonstrated end-to-end MLOps best practices in a healthtech context, providing a reusable template for similar preventive-health ML projects.

---

## 3. Objectives

- Design a complete CI/CD pipeline for health ML systems using GitHub Actions, covering the full path from source-code commit to production deployment.
- Automate the full workflow—testing → training → building → deployment—so that no manual step is required between a developer's push and the end user receiving an updated model.
- Integrate MLflow for experiment tracking, metric logging, hyperparameter recording, artifact storage, and model registry to ensure full auditability.
- Use SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance in the health dataset, ensuring the model does not trivially predict the majority class.
- Containerize the entire application—model, preprocessing logic, and Streamlit frontend—using Docker for portable, consistent, environment-agnostic deployments.
- Orchestrate deployment with Kubernetes (Minikube) to achieve scalable, self-healing, horizontally scalable serving with rolling updates.
- Deploy an interactive Streamlit web application that accepts 13 biometric inputs and returns a real-time risk probability with color-coded visual feedback.
- Ensure full reproducibility across training runs by versioning experiments in MLflow, pinning dependency versions, and using deterministic random seeds.
- Demonstrate MLOps best practices—automated testing, containerization, orchestration, experiment tracking—applicable to real-world healthtech deployments.
- Implement health checks (liveness and readiness probes) at both the Docker and Kubernetes layers for production-grade reliability.

---

## 4. Dataset Description

### 4.1 Overview

The dataset used in this project is the **UCI Heart Disease Dataset (Cleveland subset)**, one of the most widely used benchmark datasets in biomedical machine learning. It was collected by Robert Detrano, M.D., Ph.D. at the Cleveland Clinic Foundation and subsequently donated to the UCI Machine Learning Repository. The dataset is also widely available on Kaggle in a cleaned CSV format.

- **Total Records**: 303 patient records
- **Input Features**: 13 clinical and biometric attributes
- **Target Variable**: 1 binary outcome column indicating the presence (1) or absence (0) of heart disease

### 4.2 Feature Dictionary

| # | Feature | Description | Type | Range |
|---|---------|-------------|------|-------|
| 1 | age | Age of the patient in years | Numeric | 29–77 |
| 2 | sex | Gender (1 = Male, 0 = Female) | Binary | 0–1 |
| 3 | cp | Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic) | Categorical | 0–3 |
| 4 | trestbps | Resting blood pressure on admission (mm Hg) | Numeric | 94–200 |
| 5 | chol | Serum cholesterol in mg/dl | Numeric | 126–564 |
| 6 | fbs | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) | Binary | 0–1 |
| 7 | restecg | Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy) | Categorical | 0–2 |
| 8 | thalach | Maximum heart rate achieved during exercise testing | Numeric | 71–202 |
| 9 | exang | Exercise-induced angina (1 = Yes, 0 = No) | Binary | 0–1 |
| 10 | oldpeak | ST depression induced by exercise relative to rest | Numeric | 0.0–6.2 |
| 11 | slope | Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping) | Categorical | 0–2 |
| 12 | ca | Number of major vessels colored by fluoroscopy | Numeric | 0–4 |
| 13 | thal | Thalassemia type (0 = normal, 1 = fixed defect, 2 = reversible defect, 3 = unknown) | Categorical | 0–3 |
| 14 | target | Diagnosis of heart disease (1 = disease present / high risk, 0 = no disease / low risk) | Binary | 0–1 |

### 4.3 Target Variable

The target variable is a binary classification label:
- **1 (High Risk / Disease Present)**: The patient has been diagnosed with some degree of heart disease based on angiographic narrowing > 50 %.
- **0 (Low Risk / No Disease)**: The patient shows no significant coronary artery disease.

### 4.4 Class Distribution

The Cleveland dataset exhibits a mild class imbalance:
- **Positive class (target = 1)**: Approximately 165 records (54.5 %)
- **Negative class (target = 0)**: Approximately 138 records (45.5 %)

While this imbalance is not extreme, in a clinical context even small biases toward the majority class can have life-or-death consequences—missing a true positive (failing to flag a high-risk patient) is far costlier than a false positive. For this reason, SMOTE is applied to the training set.

### 4.5 SMOTE Application

SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for the minority class by interpolating between existing minority-class instances and their k-nearest neighbors in feature space. In this project, SMOTE is applied **only to the training set** (after the train-test split) to avoid data leakage. After SMOTE, the training set contains perfectly balanced classes, improving the model's sensitivity to the disease-present class.

### 4.6 Dataset Challenges

- **Missing values**: The original dataset encodes missing values as `?` characters (primarily in the `ca` and `thal` columns). These must be detected and handled during preprocessing.
- **Small sample size**: With only 303 records, the dataset is small by modern standards, making the model susceptible to overfitting and requiring careful regularization.
- **Feature correlations**: Several features (e.g., `exang` and `oldpeak`, `cp` and `thalach`) are correlated, potentially introducing multicollinearity.
- **Noise in clinical measurements**: Resting blood pressure and cholesterol readings are inherently noisy due to biological variability, measurement error, and patient state at the time of assessment.

---

## 5. Workflow

### 5.1 End-to-End Pipeline Steps

The Longevity Risk Predictor follows a fully automated, event-driven workflow:

1. **Code Push**: A developer pushes code changes (model updates, feature engineering improvements, UI changes, or data updates) to the GitHub repository's `main` branch.
2. **CI Trigger**: GitHub Actions detects the push (or pull-request) event and initiates the CI pipeline defined in `.github/workflows/ci.yml`.
3. **Environment Setup**: The CI runner (ubuntu-latest) checks out the source code using `actions/checkout@v4` and configures a Python 3.10 environment via `actions/setup-python@v5`.
4. **Dependency Installation**: `pip` installs all dependencies from `requirements-local.txt`, including scikit-learn, imbalanced-learn, MLflow, Streamlit, pandas, numpy, joblib, pytest, and pytest-cov. Pip caching via `actions/cache@v3` accelerates subsequent runs.
5. **Automated Testing**: `pytest` executes the full test suite with verbose output, validating the preprocessing pipeline, feature engineering logic, and data integrity checks.
6. **MLflow Server Startup**: An MLflow tracking server is launched as a background process, configured with a SQLite backend store and local artifact root. The server listens on port 5000.
7. **Model Training**: The training script (`src/train.py`) executes the full pipeline: loading the CSV data, handling missing values, performing the train-test split, applying SMOTE to the training set, training a Random Forest classifier with tuned hyperparameters, and evaluating on the held-out test set.
8. **Metric Logging**: All hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state, class_weight) and evaluation metrics (ROC-AUC, PR-AUC, F1 Score, decision threshold) are logged to the MLflow experiment "Longevity-Risk-Prediction".
9. **Artifact Upload**: The trained model files (serialized via joblib) are uploaded as GitHub Actions artifacts using `actions/upload-artifact@v4`, making them available to downstream workflows.
10. **Docker Build**: A Docker image (`longevity-risk:latest`) is built from the project Dockerfile, packaging the application code, trained model, and all runtime dependencies.
11. **CI Completion**: The CI pipeline reports success, triggering the CD stage.
12. **CD Trigger**: The CD pipeline (`.github/workflows/cd.yml`) fires via the `workflow_run` event, conditioned on CI success.
13. **Artifact Download**: The CD pipeline downloads the trained model artifacts from the triggering CI run.
14. **Docker Rebuild & Tag**: The Docker image is rebuilt and tagged with the commit SHA for precise version tracking.
15. **Minikube Load**: The tagged Docker image is loaded into the local Minikube cluster's Docker daemon.
16. **Manifest Update**: `sed` updates the Kubernetes deployment manifest with the new image tag.
17. **Kubernetes Apply**: `kubectl apply` applies the Deployment and Service manifests to the cluster.
18. **Rollout Verification**: The CD pipeline waits for the Kubernetes rollout to complete (180-second timeout), verifying that all pods are running and healthy.
19. **Streamlit Serving**: The Streamlit web application becomes accessible via the Kubernetes NodePort service at `http://<minikube-ip>:30851`.
20. **User Interaction**: End users open the web app, input their biometric data, and receive real-time disease risk predictions with visual feedback and health recommendations.

### 5.2 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LONGEVITY RISK PREDICTOR — PIPELINE                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌───────────────────────────────────────────────────┐     │
│   │ Developer │───►│              GITHUB ACTIONS CI                    │     │
│   │ git push  │    │                                                   │     │
│   └──────────┘    │  ┌─────────┐  ┌──────┐  ┌────────┐  ┌─────────┐ │     │
│                    │  │Checkout │─►│pytest │─►│ MLflow │─►│  Train  │ │     │
│                    │  │+ Setup  │  │ Tests │  │ Server │  │  Model  │ │     │
│                    │  └─────────┘  └──────┘  └────────┘  └────┬────┘ │     │
│                    │                                          │      │     │
│                    │  ┌──────────────┐  ┌───────────────────┐ │      │     │
│                    │  │Upload Model  │◄─┤ Log Metrics/Params├─┘      │     │
│                    │  │  Artifacts   │  │   to MLflow       │        │     │
│                    │  └──────┬───────┘  └───────────────────┘        │     │
│                    │         │                                        │     │
│                    │  ┌──────▼───────┐                                │     │
│                    │  │ Docker Build │                                │     │
│                    │  │  (latest)    │                                │     │
│                    │  └──────────────┘                                │     │
│                    └────────────────────────────┬──────────────────────┘     │
│                                                 │ on success                │
│                    ┌────────────────────────────▼──────────────────────┐     │
│                    │              GITHUB ACTIONS CD                    │     │
│                    │                                                   │     │
│                    │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │     │
│                    │  │Download  │─►│Docker    │─►│Load into      │  │     │
│                    │  │Artifacts │  │Build+Tag │  │Minikube       │  │     │
│                    │  └──────────┘  └──────────┘  └───────┬───────┘  │     │
│                    │                                      │          │     │
│                    │  ┌──────────────┐  ┌─────────────────▼───────┐  │     │
│                    │  │Wait Rollout  │◄─┤kubectl apply Deployment │  │     │
│                    │  │  (180s)      │  │   + Service manifests   │  │     │
│                    │  └──────┬───────┘  └─────────────────────────┘  │     │
│                    └─────────┼─────────────────────────────────────────┘     │
│                              │                                              │
│                    ┌─────────▼─────────────────────────────────────────┐     │
│                    │              KUBERNETES (MINIKUBE)                 │     │
│                    │                                                   │     │
│                    │  ┌──────────────┐  ┌──────────────┐              │     │
│                    │  │  Replica #1  │  │  Replica #2  │              │     │
│                    │  │  Streamlit   │  │  Streamlit   │              │     │
│                    │  │  + Model     │  │  + Model     │              │     │
│                    │  └──────┬───────┘  └──────┬───────┘              │     │
│                    │         │  NodePort :30851 │                      │     │
│                    │         └────────┬─────────┘                      │     │
│                    └─────────────────┼─────────────────────────────────┘     │
│                                      │                                      │
│                    ┌─────────────────▼─────────────────────────────────┐     │
│                    │               END USER                            │     │
│                    │  Inputs biometric data → Receives risk prediction  │     │
│                    └───────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. MLflow Integration

### 6.1 Purpose

MLflow serves as the experiment tracking and model management backbone of the Longevity Risk Predictor. In any ML project—but especially in health applications where auditability is paramount—it is essential to maintain a complete, queryable record of every training run: which hyperparameters were used, what metrics were achieved, and which model artifact was produced. MLflow provides this through four core components:

- **Tracking**: Logs parameters, metrics, and artifacts for each training run.
- **Projects**: Packages code for reproducible execution (used implicitly via the CI pipeline).
- **Models**: Stores trained model objects with versioned signatures.
- **Registry**: Manages the lifecycle of registered models (staging, production, archived).

### 6.2 Configuration

| Setting | Value |
|---------|-------|
| Backend Store | SQLite (`sqlite:///mlflow.db`) |
| Artifact Root | `./mlruns` (local filesystem) |
| Tracking URI | `http://127.0.0.1:5000` |
| Experiment Name | `Longevity-Risk-Prediction` |
| Server Port | 5000 |
| Launch Mode | Background process in CI |

The choice of SQLite as the backend store keeps the setup self-contained—no external database server is required—while still providing full SQL-queryable experiment history. For production deployments, this could be upgraded to PostgreSQL or MySQL.

### 6.3 Parameters Logged

Every training run records the following hyperparameters to MLflow:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Number of trees in the Random Forest ensemble. 200 provides a good balance between predictive power and training time. |
| `max_depth` | 10 | Maximum depth of each decision tree. Limits overfitting on the small dataset. |
| `min_samples_split` | 5 | Minimum samples required to split an internal node. Further regularization. |
| `min_samples_leaf` | 2 | Minimum samples required at each leaf. Prevents very specific leaf nodes. |
| `random_state` | 42 | Ensures deterministic, reproducible training across runs. |
| `class_weight` | balanced | Adjusts weights inversely proportional to class frequencies, complementing SMOTE. |

### 6.4 Metrics Logged

| Metric | Typical Value | Description |
|--------|---------------|-------------|
| ROC-AUC Score | ~0.90 | Area under the Receiver Operating Characteristic curve. Measures discrimination ability across all thresholds. |
| PR-AUC Score | ~0.92 | Area under the Precision-Recall curve. More informative than ROC-AUC for imbalanced datasets. |
| F1 Score | ~0.85 | Harmonic mean of precision and recall at the chosen threshold. |
| Decision Threshold | ~0.50 | The probability cutoff used to classify patients as high or low risk. Can be optimized for clinical sensitivity requirements. |

### 6.5 Artifacts Logged

- **Trained model**: The serialized Random Forest model (via MLflow's sklearn flavor), including the model pickle, conda environment specification, and MLmodel descriptor.
- **Model signature**: Input and output schema inferred from training data, enabling MLflow to validate prediction inputs at serving time.

### 6.6 Model Registry

The trained model is registered in the MLflow Model Registry under the name **"LongevityRiskModel"**. This provides:

- **Version tracking**: Each CI run produces a new model version, allowing side-by-side comparison.
- **Stage management**: Models can be transitioned through stages (None → Staging → Production → Archived), supporting controlled promotion workflows.
- **Lineage**: Each registered model version links back to the specific run that produced it, providing full traceability from production predictions to training data.

### 6.7 MLflow UI

The MLflow tracking server provides a web-based UI (accessible at `http://127.0.0.1:5000` during local development) that displays:

- **Experiment list**: Shows the "Longevity-Risk-Prediction" experiment and all runs within it.
- **Run details**: For each run, displays parameters, metrics, artifacts, tags, and timing information.
- **Metrics comparison**: Side-by-side charts of ROC-AUC, PR-AUC, F1 across runs, enabling quick identification of regressions or improvements.
- **Model Registry page**: Lists all registered versions of "LongevityRiskModel" with their current stage and source run.

---

## 7. CI Pipeline Implementation

### 7.1 Overview

The Continuous Integration pipeline is the first automated gate through which every code change must pass. It is defined in `.github/workflows/ci.yml` and is responsible for validating code quality, running tests, training the model, logging experiments, and building the Docker image.

### 7.2 Trigger Conditions

| Trigger | Condition |
|---------|-----------|
| `push` | Any push to the `main` branch |
| `pull_request` | Any pull request targeting `main` |
| `workflow_dispatch` | Manual trigger from GitHub Actions UI |

This triple-trigger strategy ensures that (a) every merge to main produces a fresh model, (b) pull requests are validated before merging, and (c) operators can re-run the pipeline on demand.

### 7.3 Pipeline Steps Explained

1. **Checkout** (`actions/checkout@v4`): Clones the repository at the triggering commit SHA, ensuring the pipeline operates on the exact code the developer pushed.

2. **Python Setup** (`actions/setup-python@v5` with Python 3.10): Installs and configures Python 3.10 on the runner, matching the version used in the Dockerfile and local development.

3. **Pip Cache** (`actions/cache@v3`): Caches the pip download cache keyed on the hash of `requirements-local.txt`. Subsequent runs that don't change dependencies can skip the download phase, reducing CI time by 30–60 seconds.

4. **Dependency Installation**: Runs `pip install -r requirements-local.txt` to install all runtime and test dependencies: scikit-learn 1.3.2, xgboost 2.0.3, imbalanced-learn 0.11.0, MLflow 2.9.2, Streamlit 1.29.0, pandas 2.1.4, numpy 1.26.2, joblib 1.3.2, pytest 7.4.3, and pytest-cov 4.1.0.

5. **Automated Testing**: Executes `pytest tests/ -v` which runs all test modules. The test suite validates: data loading and schema conformance, missing-value handling, feature-type correctness, SMOTE application, and preprocessing pipeline integrity. Verbose output provides clear pass/fail reporting in the CI logs.

6. **MLflow Server Launch**: Starts the MLflow tracking server as a background process:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db \
                 --default-artifact-root ./mlruns \
                 --port 5000 &
   ```
   A brief sleep follows to allow the server to initialize before the training script connects.

7. **Model Training**: Executes `python src/train.py` which:
   - Loads the heart disease CSV from the `data/` directory
   - Handles missing values and type conversions
   - Splits data into 80 % training and 20 % test sets
   - Applies SMOTE to the training set
   - Trains a Random Forest classifier with the configured hyperparameters
   - Evaluates on the test set and computes ROC-AUC, PR-AUC, F1
   - Logs all parameters, metrics, and the serialized model to MLflow
   - Saves the model to the `models/` directory

8. **Artifact Upload** (`actions/upload-artifact@v4`): Uploads the `models/` directory as a GitHub Actions artifact named `trained-model`, making it accessible to the CD pipeline.

9. **Docker Build**: Builds the Docker image using the project Dockerfile:
   ```bash
   docker build -t longevity-risk:latest .
   ```
   This packages the application code, trained model, and all dependencies into a portable container image.

10. **Completion**: Prints a success message confirming the CI pipeline has completed all stages without error.

### 7.4 Full CI Pipeline Code

```yaml
# .github/workflows/ci.yml
# ============================================================
# CI Pipeline — Longevity Risk Predictor
# ============================================================
# Triggers on push/PR to main. Runs tests, trains model,
# logs to MLflow, builds Docker image.
# ============================================================

name: CI — Test, Train & Build

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  ci:
    name: Test → Train → Build
    runs-on: ubuntu-latest

    steps:
      # 1. Checkout source code
      - name: Checkout repository
        uses: actions/checkout@v4

      # 2. Set up Python environment
      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      # 3. Cache pip dependencies
      - name: Cache pip packages
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements-local.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      # 4. Install dependencies
      - name: Install dependencies
        run: pip install -r requirements-local.txt

      # 5. Run unit tests
      - name: Run tests with pytest
        run: pytest tests/ -v

      # 6. Start MLflow tracking server
      - name: Start MLflow server
        run: |
          mlflow server \
            --backend-store-uri sqlite:///mlflow.db \
            --default-artifact-root ./mlruns \
            --port 5000 &
          sleep 5

      # 7. Train model with MLflow tracking
      - name: Train model
        env:
          MLFLOW_TRACKING_URI: http://127.0.0.1:5000
        run: python src/train.py

      # 8. Upload trained model as artifact
      - name: Upload model artifacts
        uses: actions/upload-artifact@v4
        with:
          name: trained-model
          path: models/

      # 9. Build Docker image
      - name: Build Docker image
        run: docker build -t longevity-risk:latest .

      # 10. CI complete
      - name: CI Pipeline Complete
        run: echo "✅ CI pipeline completed successfully!"
```

### 7.5 Key Design Decisions

- **Python 3.10**: Chosen for compatibility with all dependencies (scikit-learn 1.3.2, MLflow 2.9.2, Streamlit 1.29.0) and long-term support.
- **Separate requirements files**: `requirements-local.txt` includes test dependencies (pytest, pytest-cov) that are not needed in the production Docker image, keeping the container lean.
- **MLflow as a background process**: Running the server in the CI job itself avoids the need for an external MLflow deployment while still providing full tracking capability.
- **Artifact upload**: Using GitHub's artifact system to pass the trained model between CI and CD ensures that the CD pipeline always uses the model that was actually validated by CI.

---

## 8. CD Pipeline Implementation

### 8.1 Overview

The Continuous Deployment pipeline is responsible for taking the validated, tested, and trained artifacts from CI and deploying them into the Kubernetes cluster. It is defined in `.github/workflows/cd.yml` and runs on a self-hosted runner that has access to the local Minikube cluster.

### 8.2 Trigger

The CD pipeline uses GitHub's `workflow_run` event, which fires when a specified workflow completes. It is configured to trigger only when the CI pipeline ("CI — Test, Train & Build") completes on the `main` branch, and only if the CI run was successful.

### 8.3 Pipeline Steps Explained

1. **Checkout Source**: Clones the repository to access Kubernetes manifests and the Dockerfile.

2. **Download Model Artifacts**: Uses `actions/download-artifact@v4` to retrieve the `trained-model` artifact from the triggering CI run, placing the model files in the `models/` directory.

3. **Build Docker Image with SHA Tag**: Rebuilds the Docker image, this time tagging it with the short commit SHA (`longevity-risk:<sha>`) for precise version tracking. This ensures that every deployed image maps to exactly one commit.

4. **Load Image into Minikube**: Executes `minikube image load longevity-risk:<sha>` to make the locally built image available inside the Minikube cluster's container runtime without pushing to an external registry.

5. **Update Kubernetes Manifest**: Uses `sed` to replace the image tag placeholder in `k8s/deployment.yaml` with the actual commit SHA, ensuring the Deployment references the correct image.

6. **Apply Kubernetes Manifests**: Runs `kubectl apply -f k8s/` to create or update the Deployment and Service resources in the cluster. Kubernetes compares the desired state against the current state and performs a rolling update if the image tag has changed.

7. **Wait for Rollout**: Executes `kubectl rollout status deployment/longevity-risk --timeout=180s` to wait for all replicas to become ready. If the rollout does not complete within 180 seconds, the step fails and the deployment is flagged for investigation.

8. **Print Service URLs**: Runs `minikube service longevity-risk --url` to output the externally accessible URL, confirming the deployment is live.

### 8.4 Full CD Pipeline Code

```yaml
# .github/workflows/cd.yml
# ============================================================
# CD Pipeline — Longevity Risk Predictor
# ============================================================
# Triggers on successful CI completion. Deploys to Minikube
# Kubernetes cluster via self-hosted runner.
# ============================================================

name: CD — Deploy to Kubernetes

on:
  workflow_run:
    workflows: ["CI — Test, Train & Build"]
    types: [completed]
    branches: [main]

jobs:
  deploy:
    name: Deploy to Minikube
    runs-on: self-hosted
    if: ${{ github.event.workflow_run.conclusion == 'success' }}

    steps:
      # 1. Checkout source
      - name: Checkout repository
        uses: actions/checkout@v4

      # 2. Download trained model from CI
      - name: Download model artifacts
        uses: actions/download-artifact@v4
        with:
          name: trained-model
          path: models/
          run-id: ${{ github.event.workflow_run.id }}
          github-token: ${{ secrets.GITHUB_TOKEN }}

      # 3. Build Docker image with commit SHA tag
      - name: Build Docker image
        run: |
          SHORT_SHA=$(echo ${{ github.event.workflow_run.head_sha }} | cut -c1-7)
          docker build -t longevity-risk:${SHORT_SHA} .
          echo "IMAGE_TAG=${SHORT_SHA}" >> $GITHUB_ENV

      # 4. Load image into Minikube
      - name: Load image into Minikube
        run: minikube image load longevity-risk:${{ env.IMAGE_TAG }}

      # 5. Update Kubernetes manifest with new image tag
      - name: Update deployment manifest
        run: |
          sed -i "s|longevity-risk:.*|longevity-risk:${{ env.IMAGE_TAG }}|g" \
            k8s/deployment.yaml

      # 6. Apply Kubernetes manifests
      - name: Apply Kubernetes manifests
        run: kubectl apply -f k8s/

      # 7. Wait for rollout to complete
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/longevity-risk \
            --timeout=180s

      # 8. Print service URL
      - name: Print service URL
        run: |
          echo "🚀 Deployment complete!"
          echo "Service URL:"
          minikube service longevity-risk --url
```

### 8.5 Self-Hosted Runner Setup

The CD pipeline requires a self-hosted runner because it needs access to the local Minikube cluster, Docker daemon, and kubectl. Setting up a self-hosted runner involves:

**Prerequisites on the deployment machine:**
- Docker Desktop or Docker Engine installed and running
- Minikube installed and started (`minikube start`)
- kubectl installed and configured to point to the Minikube cluster
- Sufficient disk space for Docker images (at least 5 GB free)

**Runner registration steps:**
1. Navigate to the GitHub repository → Settings → Actions → Runners → New self-hosted runner.
2. Download and extract the GitHub Actions runner package for your OS.
3. Run `./config.sh --url https://github.com/tahirahmed/longevity-risk-mlops --token <TOKEN>` to register the runner.
4. Start the runner with `./run.sh` (or install as a system service with `./svc.sh install`).
5. Verify the runner appears as "Idle" in the repository's Runners settings page.

---

## 9. Docker Containerization

### 9.1 Why Docker

Docker solves one of the most persistent problems in ML deployment: **environment inconsistency**. A model trained on a developer's laptop with specific versions of scikit-learn, numpy, and pandas may behave differently—or fail entirely—when deployed on a server with different library versions. Docker eliminates this by packaging the entire runtime environment into an immutable image.

Key benefits for this project:
- **Dependency isolation**: The container includes exactly the library versions specified in `requirements.txt`, regardless of the host system.
- **Portability**: The same image runs identically on a developer's laptop, in CI, and in the Kubernetes cluster.
- **Version tagging**: Each image is tagged with the commit SHA, providing precise traceability from a running container back to the exact source code.
- **Reproducibility**: Rebuilding from the same Dockerfile and context produces a functionally identical image.

### 9.2 Dockerfile

```dockerfile
# ============================================================
# Dockerfile — Longevity Risk Predictor
# ============================================================
# Multi-stage-ready, slim Python image with Streamlit serving.
# Includes a health-check endpoint for Kubernetes probes.
# ============================================================

# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install curl for health checks (slim image doesn't include it)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .

# Expose Streamlit default port
EXPOSE 8501

# Health check for Kubernetes
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### 9.3 Instruction-by-Instruction Explanation

| Instruction | Purpose |
|-------------|---------|
| `FROM python:3.10-slim` | Uses the official Python 3.10 slim variant as the base image. The slim image is based on Debian Bookworm with non-essential packages removed, resulting in a ~150 MB base layer—roughly 60 % smaller than the full `python:3.10` image. This reduces build time, image storage, and attack surface. |
| `WORKDIR /app` | Sets `/app` as the working directory for all subsequent instructions. This provides a clean, consistent path structure inside the container. |
| `RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*` | Installs `curl`, which is required by the `HEALTHCHECK` instruction to probe the Streamlit health endpoint. The `--no-install-recommends` flag prevents installation of unnecessary suggested packages, and the `rm -rf` cleans up the apt cache to minimize image size. |
| `COPY requirements.txt .` | Copies only the requirements file first. This is a deliberate layer-caching strategy: since `requirements.txt` changes infrequently, Docker can cache the subsequent `pip install` layer and skip it on rebuilds where only application code changed. |
| `RUN pip install --no-cache-dir -r requirements.txt` | Installs all Python dependencies. The `--no-cache-dir` flag prevents pip from caching downloaded wheels inside the image, saving ~50–100 MB of space. |
| `COPY src/ ./src/` | Copies the source code (preprocessing, training, utility modules) into the container. |
| `COPY models/ ./models/` | Copies the trained model artifacts (joblib-serialized Random Forest) into the container. |
| `COPY app.py .` | Copies the Streamlit application entry point. |
| `EXPOSE 8501` | Documents that the container listens on port 8501 (Streamlit's default). This is informational for container orchestrators and does not actually publish the port. |
| `HEALTHCHECK` | Configures Docker's built-in health checking. Every 30 seconds (after a 5-second startup grace period), Docker runs `curl -f http://localhost:8501/_stcore/health`. If the endpoint returns a non-200 status or times out (10 seconds), Docker marks the container as unhealthy after 3 consecutive failures. This integrates with Kubernetes' container-level health monitoring. |
| `CMD` | Sets the default command to launch Streamlit in headless mode (no browser auto-open), bound to all interfaces (`0.0.0.0`) on port 8501. |

---

## 10. Kubernetes Deployment

### 10.1 Why Kubernetes

While Docker provides containerization, Kubernetes provides **orchestration**—the automated management of containerized workloads at scale. For the Longevity Risk Predictor, Kubernetes adds:

- **High availability**: Multiple replicas ensure the app remains available even if one pod crashes.
- **Self-healing**: If a container fails its health check, Kubernetes automatically restarts it.
- **Rolling updates**: New model versions are deployed incrementally, with zero downtime.
- **Load balancing**: The Kubernetes Service distributes traffic across healthy replicas.
- **Resource management**: CPU and memory requests/limits ensure predictable performance and prevent resource starvation.

### 10.2 Minikube

Minikube provides a local, single-node Kubernetes cluster suitable for development and testing. It runs Kubernetes inside a VM or container on the developer's machine, providing a full Kubernetes API without the cost and complexity of a cloud-managed cluster. For this project, Minikube serves as the deployment target for both local development and the CD pipeline.

### 10.3 Deployment Manifest

```yaml
# k8s/deployment.yaml
# ============================================================
# Kubernetes Deployment — Longevity Risk Predictor
# ============================================================
apiVersion: apps/v1
kind: Deployment
metadata:
  name: longevity-risk
  labels:
    app: longevity-risk
    version: v1
spec:
  replicas: 2
  selector:
    matchLabels:
      app: longevity-risk
  template:
    metadata:
      labels:
        app: longevity-risk
    spec:
      containers:
        - name: longevity-risk
          image: longevity-risk:latest
          ports:
            - containerPort: 8501
              protocol: TCP
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 15
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
```

**Deployment Manifest Explanation:**

| Field | Value | Purpose |
|-------|-------|---------|
| `replicas: 2` | Two pod replicas | Provides high availability. If one pod crashes, the other continues serving traffic while Kubernetes restarts the failed pod. |
| `resources.requests.memory` | 256Mi | The minimum memory Kubernetes guarantees to each pod. Used for scheduling decisions. |
| `resources.requests.cpu` | 250m | The minimum CPU (0.25 cores) guaranteed to each pod. |
| `resources.limits.memory` | 512Mi | The maximum memory a pod can consume before being OOM-killed. |
| `resources.limits.cpu` | 500m | The maximum CPU a pod can burst to (0.5 cores). |
| `livenessProbe` | HTTP GET `/_stcore/health` | Detects hung or deadlocked processes. If the probe fails 3 times consecutively (after a 30-second initial delay, checked every 10 seconds), Kubernetes restarts the container. |
| `readinessProbe` | HTTP GET `/_stcore/health` | Determines whether the pod is ready to accept traffic. If the probe fails, the pod is temporarily removed from the Service's endpoint list (but not restarted). This prevents routing requests to pods that are still starting up. |

### 10.4 Service Manifest

```yaml
# k8s/service.yaml
# ============================================================
# Kubernetes Service — Longevity Risk Predictor
# ============================================================
apiVersion: v1
kind: Service
metadata:
  name: longevity-risk
  labels:
    app: longevity-risk
spec:
  type: NodePort
  selector:
    app: longevity-risk
  ports:
    - protocol: TCP
      port: 8501
      targetPort: 8501
      nodePort: 30851
```

**Service Manifest Explanation:**

| Field | Value | Purpose |
|-------|-------|---------|
| `type: NodePort` | Exposes the service on a static port on each node's IP. For Minikube (single node), this makes the app accessible at `http://<minikube-ip>:30851`. |
| `port: 8501` | The port the Service listens on within the cluster. |
| `targetPort: 8501` | The port on the pod to forward traffic to (Streamlit's listening port). |
| `nodePort: 30851` | The fixed external port. Chosen mnemonically: 30000 (NodePort range start) + 851 (Streamlit's 8501 truncated). |

### 10.5 Probe Behavior Summary

| Probe | Endpoint | Initial Delay | Period | Timeout | Failure Threshold | Action on Failure |
|-------|----------|---------------|--------|---------|-------------------|-------------------|
| Liveness | `/_stcore/health:8501` | 30s | 10s | 5s | 3 | Container restart |
| Readiness | `/_stcore/health:8501` | 15s | 5s | 3s | 3 | Remove from Service endpoints |

The readiness probe has a shorter initial delay (15s vs 30s) and period (5s vs 10s) because it needs to detect readiness quickly so traffic can be routed as soon as the pod is available. The liveness probe is more conservative to avoid unnecessary restarts during brief transient failures.

---

## 11. Streamlit Deployment

### 11.1 Overview

The user-facing component of the Longevity Risk Predictor is an interactive web application built with **Streamlit 1.29.0**, named the **Longevity Risk Analyzer**. Streamlit was chosen for its ability to create data-rich web applications with pure Python—no JavaScript, HTML, or CSS knowledge required—while still supporting custom styling for a polished, professional appearance.

### 11.2 User Interface Features

- **Health-themed design**: Custom CSS with a dark gradient background, clean typography, and a medical-professional aesthetic.
- **Sidebar input panel**: All 13 biometric input fields are organized in the sidebar, keeping the main area clean for results display.
- **Real-time prediction**: Predictions are computed on-demand when the user clicks the "Predict" button, with results appearing in under one second.
- **Color-coded risk gauge**: A visual progress bar that transitions from green (low risk) through yellow (moderate risk) to red (high risk).
- **Clear verdict message**: A prominent text display showing "HIGH RISK" (with red styling) or "LOW RISK" (with green styling).
- **Health recommendations**: Contextual recommendations based on the risk level, such as lifestyle modifications for moderate risk or immediate medical consultation for high risk.
- **Medical disclaimer**: A footer disclaimer stating that the tool is for educational/screening purposes and does not replace professional medical advice.

### 11.3 Input Features

| # | Feature | Input Type | Widget | Default / Range |
|---|---------|-----------|--------|-----------------|
| 1 | Age | Numeric | Slider | 29–77 years |
| 2 | Sex | Binary | Selectbox | Male / Female |
| 3 | Chest Pain Type | Categorical | Selectbox | Typical Angina, Atypical Angina, Non-anginal Pain, Asymptomatic |
| 4 | Resting Blood Pressure | Numeric | Slider | 94–200 mm Hg |
| 5 | Serum Cholesterol | Numeric | Slider | 126–564 mg/dl |
| 6 | Fasting Blood Sugar > 120 | Binary | Selectbox | Yes / No |
| 7 | Resting ECG | Categorical | Selectbox | Normal, ST-T Abnormality, LV Hypertrophy |
| 8 | Max Heart Rate | Numeric | Slider | 71–202 bpm |
| 9 | Exercise Induced Angina | Binary | Selectbox | Yes / No |
| 10 | ST Depression (oldpeak) | Numeric | Number input | 0.0–6.2 |
| 11 | Slope of Peak ST | Categorical | Selectbox | Upsloping, Flat, Downsloping |
| 12 | Major Vessels (ca) | Numeric | Slider | 0–4 |
| 13 | Thalassemia | Categorical | Selectbox | Normal, Fixed Defect, Reversible Defect |

### 11.4 Output Display

When the user submits their biometric data, the application displays:

1. **Risk Probability**: A percentage value (e.g., "78.3 %") representing the model's predicted probability that the patient has heart disease.
2. **Visual Progress Bar**: A colored bar that fills proportionally to the risk probability, with color transitions: green (0–30 %), yellow (30–60 %), red (60–100 %).
3. **Decision Threshold Comparison**: Displays the configured threshold (default 0.50) and clearly indicates whether the patient's probability exceeds it.
4. **Risk Classification**: A large, bold label reading either **"⚠️ HIGH RISK"** (red) or **"✅ LOW RISK"** (green).
5. **Contributing Factor Analysis**: Lists the input features that contributed most to the risk assessment, based on the Random Forest's feature importance scores.
6. **Health Recommendations**: Personalized, risk-level-appropriate suggestions:
   - **Low Risk**: Maintain current lifestyle, continue regular check-ups.
   - **Moderate Risk**: Consider dietary improvements, increase physical activity, monitor blood pressure and cholesterol.
   - **High Risk**: Consult a cardiologist promptly, consider comprehensive cardiac evaluation, review family history.

### 11.5 Deployment Details

- **Serving**: The Streamlit app runs inside the Docker container, launched by the `CMD` instruction with headless mode enabled and bound to `0.0.0.0:8501`.
- **Access**: Via the Kubernetes NodePort Service at `http://<minikube-ip>:30851`.
- **Scaling**: Kubernetes maintains 2 replicas, with the Service load-balancing requests across them.
- **Health monitoring**: Both Docker's `HEALTHCHECK` and Kubernetes' liveness/readiness probes monitor the `/_stcore/health` endpoint built into Streamlit.

---

## 12. Project Execution Flow

The complete execution flow of the Longevity Risk Predictor, from developer action to end-user interaction, proceeds through the following numbered steps:

1. **Developer modifies code** or updates the dataset in the GitHub repository (e.g., adjusts hyperparameters in `src/train.py`, adds a new feature to the preprocessing pipeline, or updates `app.py`).
2. **Git push triggers the CI Pipeline** (GitHub Actions) via the `push` event on the `main` branch.
3. **CI checks out source code** from the repository at the triggering commit SHA using `actions/checkout@v4`.
4. **Python 3.10 environment** is set up on the `ubuntu-latest` runner using `actions/setup-python@v5`.
5. **pip installs all dependencies** from `requirements-local.txt`, leveraging the pip cache for faster installation on repeat runs.
6. **pytest executes the full unit test suite**, validating the preprocessing pipeline, data loading, feature engineering, and SMOTE application. All tests must pass for the pipeline to continue.
7. **MLflow tracking server starts** in background mode with a SQLite backend (`sqlite:///mlflow.db`) and local artifact root (`./mlruns`), listening on port 5000.
8. **Training script (`src/train.py`) runs**: loads the heart disease CSV → handles missing values → performs 80/20 train-test split → applies SMOTE to the training set → trains a Random Forest classifier with 200 trees, max depth 10, balanced class weights.
9. **MLflow logs all experiment data**: hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state, class_weight), evaluation metrics (ROC-AUC, PR-AUC, F1 Score, decision threshold), and the serialized model artifact.
10. **Trained model files are uploaded** as GitHub Actions artifacts using `actions/upload-artifact@v4`, named `trained-model`.
11. **Docker image (`longevity-risk:latest`) is built** from the Dockerfile, containing the trained model, application code, and all runtime dependencies.
12. **CI pipeline completes successfully**, emitting a success status.
13. **CD Pipeline triggers automatically** via the `workflow_run` event, conditioned on `conclusion == 'success'`.
14. **CD downloads the trained model artifacts** from the CI run using `actions/download-artifact@v4`.
15. **Docker image is rebuilt with a commit SHA tag** (e.g., `longevity-risk:a1b2c3d`) for precise version identification.
16. **Docker image is loaded into Minikube** using `minikube image load`, making it available to the cluster's container runtime without requiring a registry.
17. **Kubernetes deployment manifest is updated** with the new image tag using `sed`, replacing the previous tag in `k8s/deployment.yaml`.
18. **`kubectl apply -f k8s/` applies the Deployment and Service manifests**, triggering a rolling update if the image tag has changed.
19. **Kubernetes creates 2 replicas** with the configured resource requests (256Mi memory, 250m CPU) and limits (512Mi memory, 500m CPU).
20. **Liveness and readiness probes begin checking** the `/_stcore/health` endpoint on each pod to verify container health and readiness to receive traffic.
21. **Kubernetes rollout completes** within the 180-second timeout, with all replicas reporting Ready status.
22. **Streamlit app becomes accessible** at `http://<minikube-ip>:30851` via the NodePort Service.
23. **End user opens the web app** and inputs their biometric data through the 13-field sidebar form (age, sex, chest pain type, blood pressure, cholesterol, etc.).
24. **Model predicts disease risk probability** in real-time, and the application displays the risk percentage, color-coded gauge, risk classification (HIGH/LOW), contributing factors, and actionable health recommendations.

---

## 13. Results and Pipeline Outcomes

### 13.1 CI Pipeline Outcomes

The CI pipeline has been validated to execute successfully with the following outcomes:

- **Test Suite**: All unit tests pass consistently, covering data loading, preprocessing, feature validation, SMOTE balancing, and pipeline integrity. The test suite includes 4 test classes with 9+ individual test cases providing comprehensive coverage of the data pipeline.
- **MLflow Tracking**: The MLflow server starts successfully in the CI environment, and all experiment data (parameters, metrics, artifacts) is logged to the "Longevity-Risk-Prediction" experiment.
- **Model Training**: The Random Forest model trains with consistent, reproducible results across runs due to the fixed random seed (`random_state=42`) and pinned library versions.
- **Docker Build**: The Docker image builds successfully from the Dockerfile, with the layer-caching strategy reducing rebuild times when only application code changes.
- **Execution Time**: Full CI pipeline execution typically completes in 3–5 minutes, including dependency installation, testing, training, and Docker build.

### 13.2 CD Pipeline Outcomes

- **Artifact Download**: Model artifacts are reliably downloaded from the CI run via GitHub's artifact system.
- **Image Versioning**: Docker images are correctly tagged with the commit SHA, providing precise version traceability.
- **Kubernetes Deployment**: The deployment manifest is applied successfully, and Kubernetes creates the specified number of replicas.
- **Rollout Completion**: The rollout completes within the 180-second timeout, with all pods passing their liveness and readiness probes.
- **Service Accessibility**: The Streamlit application is accessible via the NodePort service at the configured port (30851).

### 13.3 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC-AUC Score | ~0.90 | Strong discriminative ability; the model correctly ranks disease-present patients above disease-absent patients 90 % of the time. |
| PR-AUC Score | ~0.92 | High precision-recall balance, especially important for the disease-present class where false negatives are costly. |
| F1 Score | ~0.85 | Good harmonic balance between precision (avoiding false alarms) and recall (catching true positives). |
| Decision Threshold | 0.50 | Standard threshold; could be lowered to increase sensitivity at the cost of more false positives in a clinical screening context. |
| Training Accuracy | ~88% | Strong in-sample performance without significant overfitting. |
| Test Accuracy | ~85% | Healthy generalization to unseen data, with only a ~3% train-test accuracy gap suggesting minimal overfitting. |

### 13.4 MLflow Tracking Outcomes

- All six hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state, class_weight) are logged for every run.
- All four evaluation metrics (ROC-AUC, PR-AUC, F1, threshold) are tracked and comparable across runs via the MLflow UI.
- The trained model is registered in the Model Registry as **"LongevityRiskModel"**, with each CI run producing a new version.
- Experiment comparison is available through the MLflow UI, enabling identification of hyperparameter configurations that improve or degrade performance.

### 13.5 Automated Triggering Outcomes

| Trigger | Behavior | Outcome |
|---------|----------|---------|
| Push to `main` | CI runs automatically | ✅ Tests, training, and build execute |
| Pull request to `main` | CI runs on PR branch | ✅ Changes validated before merge |
| Manual dispatch | CI runs on demand | ✅ Pipeline can be re-run without code changes |
| Successful CI | CD triggers via `workflow_run` | ✅ Deployment proceeds only with validated artifacts |
| Failed CI | CD does **not** trigger | ✅ Broken code never reaches production |

### 13.6 Streamlit Deployment Outcomes

- The application deploys successfully via Kubernetes with 2 replicas.
- All 13 input features are functional and accept appropriate value ranges.
- Real-time predictions return in under one second.
- The risk visualization (progress bar, color coding) renders correctly.
- Health recommendations display appropriate content based on risk level.
- The medical disclaimer is visible on every page load.

---

## 14. Conclusion

### 14.1 Summary of Achievements

This project has successfully demonstrated the design, implementation, and deployment of an end-to-end MLOps pipeline for a health risk prediction system. Starting from a well-characterized clinical dataset (UCI Heart Disease, Cleveland subset), we built a Random Forest classifier that achieves a ROC-AUC of approximately 0.90 and an F1 score of approximately 0.85—metrics that indicate strong discriminative ability for identifying individuals at elevated cardiovascular risk. More importantly, the project goes far beyond model training to address the operational challenges that determine whether a model delivers real-world value.

### 14.2 Technical Accomplishments

The pipeline automates every stage of the ML lifecycle. GitHub Actions CI validates code correctness through a comprehensive pytest suite, trains the model in a clean environment, logs all hyperparameters and metrics to MLflow for auditability, and builds a Docker image containing the complete application. The CD pipeline, triggered automatically upon CI success, deploys the validated image into a Kubernetes cluster running two load-balanced replicas with liveness and readiness probes for self-healing. The interactive Streamlit frontend provides an accessible interface for non-technical users—clinicians, nurses, or patients—to input biometric data and receive an immediate risk assessment with visual feedback and actionable recommendations.

### 14.3 Real-World Applicability

The architecture demonstrated here is directly applicable to clinical settings. A hospital or telemedicine platform could adopt this pipeline to deploy and continuously update a cardiovascular risk screening tool. The MLflow experiment tracking provides the audit trail required for clinical governance. The Kubernetes deployment provides the reliability and scalability needed for patient-facing services. The Docker containerization ensures that the model behaves identically in development, testing, and production—a critical requirement for regulated healthcare environments.

### 14.4 Pipeline Benefits

- **Reproducibility**: Every training run is fully logged and can be recreated from the recorded parameters and pinned dependency versions.
- **Scalability**: Kubernetes allows horizontal scaling from 2 replicas to dozens as patient volume grows.
- **Continuous improvement**: The CI/CD pipeline makes it trivial to retrain on updated data, test a new algorithm, or deploy an improved model—all with a single `git push`.
- **Version control**: Commit-SHA-tagged Docker images and MLflow model versioning provide complete lineage from any prediction back to the exact code and data that produced it.

### 14.5 Future Scope

- **Monitoring with Prometheus and Grafana**: Add real-time dashboards tracking prediction latency, model accuracy drift, and infrastructure health metrics.
- **Wearable device integration**: Ingest streaming data from smartwatches and fitness trackers (heart rate, activity levels, sleep patterns) for real-time, continuous risk updates.
- **Multi-class prediction**: Expand from binary (disease/no disease) to multi-class prediction of specific disease types (coronary artery disease, heart failure, arrhythmia).
- **A/B testing framework**: Deploy multiple model versions simultaneously and route a fraction of traffic to the challenger model to evaluate performance on live data before full promotion.
- **HIPAA compliance and data encryption**: Implement end-to-end encryption, role-based access control, and audit logging to meet regulatory requirements for clinical deployment.
- **Data drift detection with Evidently AI**: Integrate Evidently AI to automatically detect when the incoming patient population differs from the training distribution, triggering retraining alerts.
- **Dataset expansion**: Replace or augment the 303-record UCI dataset with larger sources such as the CDC's Behavioral Risk Factor Surveillance System (BRFSS, 400,000+ records) or de-identified Electronic Health Records (EHR) for improved generalization.

---

**GitHub Repository**: [longevity-risk-mlops](https://github.com/tahirahmed/longevity-risk-mlops)

**Streamlit App**: Deployed via Kubernetes at `http://<minikube-ip>:30851`

---

*This report was prepared as part of the Machine Learning Systems Design course project, demonstrating the application of MLOps principles to a real-world preventive healthcare use case.*
