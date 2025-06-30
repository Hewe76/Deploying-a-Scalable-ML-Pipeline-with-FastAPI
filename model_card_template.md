# Model Card

> **Project**: Deploying a Scalable ML Pipeline with FastAPI  
> **Model name**: Income Classifier (≥ 50 K vs. < 50 K)  
> **Version**: 1.0 • 2025-06-30

---

## Model Details
This model is a **RandomForestClassifier** from scikit-learn (`n_estimators=200`, `max_features="sqrt"`, `bootstrap=True`, `random_state=42`).  
It predicts whether a person’s annual income exceeds \$50 000 using features from the 1994 Census Adult dataset. The model was trained and evaluated within an MLOps pipeline that includes automated data preprocessing, unit tests, continuous integration (GitHub Actions running `pytest` + `flake8`), and FastAPI for deployment.

**Key attributes**

| Attribute | Value |
|-----------|-------|
| **Framework** | scikit-learn 1.4.2 |
| **Algorithm** | Random Forest (bagged decision trees) |
| **Input schema** | 8 categorical + 6 numerical features |
| **Output** | Binary label (`1` = “≥ 50 K”, `0` = “< 50 K”) |
| **Training time** | ~45 s on 8-core CPU |
| **Hardware** | HPC node (Intel Xeon, 32 GB RAM) |

---

## Intended Use
The classifier is intended as an **educational example** for the Udacity “Machine Learning DevOps Engineer” nanodegree. It demonstrates best practices for data versioning, automated testing, continuous integration, and model serving.  
It **should not** be used for high-stakes decisions such as credit approval, hiring, or resource allocation in production without a thorough fairness and bias audit.

---

## Training Data
The training data comes from the **UCI Adult (Census Income) dataset** (48 842 rows after cleanup). We used an 80 % train / 20 % test split stratified on the income label to preserve class balance.

* **Features (14 total)**  
  *Categorical*: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`  
  *Numerical*: `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`, `fnlwgt`

Categorical variables were one-hot encoded; the target label was binarized (`1` = “> 50 K”, `0` = “≤ 50 K”).

---

## Evaluation Data
The **test set** contains 9 769 rows (20 % of the original data), withheld during training. All preprocessing transformations learned on the training data (one-hot encoder, label binarizer) were applied to this test set without re-fitting.

---

## Metrics
We evaluate the model on **precision, recall, and F1-score** (β = 1). Results on the overall test set:

| Metric | Score |
|--------|-------|
| **Precision** | **0.734** |
| **Recall** | **0.637** |
| **F1-score** | **0.682** |

### Slice performance  
To assess fairness, we calculated metrics on slices of the test data defined by each unique value in the eight categorical features. For example, precision on the `sex = Female` slice was 0.72 versus 0.74 for `sex = Male`. Full per-slice results are logged in `slice_output.txt`.

---

## Ethical Considerations
Census data reflect historical socioeconomic patterns that embed **systemic biases** (e.g., gender wage gaps, racial disparities). Consequently, the model may:

* **Over-predict low income** for groups historically under-represented in high-income brackets.  
* **Reinforce past inequities** if used in automated decision-making.

Mitigations applied:

1. **Slice metrics** are reported so stakeholders can inspect disparate performance.  
2. **Documentation** (this card) explicitly warns against unintended high-stakes deployment.  
3. **Open-source code** allows independent audits and enhancements (e.g., re-sampling, fairness constraints).

More thorough bias analysis—such as equalized odds or demographic parity—was not conducted in this educational iteration but is strongly recommended for any real-world use.

---

## Caveats and Recommendations
* The model is trained on 1994 U.S. census data; **predictive accuracy may degrade** on contemporary or non-U.S. populations.  
* The pipeline assumes the same feature schema at inference time. Changes (new categories, missing values) require re-training or robust preprocessing.  
* Hyper-parameters were not exhaustively tuned; performance could be improved with cross-validated grid search or more sophisticated algorithms (e.g., Gradient Boosting, XGBoost).  
* Before deploying in production, perform a full ML-Ops checklist: monitoring, data drift detection, security hardening, and a detailed fairness assessment.

---

