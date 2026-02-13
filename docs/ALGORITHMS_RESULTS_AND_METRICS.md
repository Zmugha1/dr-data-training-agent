# Algorithms, Results, and Metrics - Evaluation Summary

This document summarizes **all algorithm work** in the Human-in-the-Loop Decision Intelligence agent pipeline so you can evaluate whether results are credible. It covers: (1) complete algorithm specification, (2) inputs and data generation, (3) processing steps, (4) metrics reported, and (5) credibility considerations.

---

## 1. Pipeline Overview

| Stage | Component | Algorithm / Method | Purpose |
|-------|-----------|---------------------|---------|
| 1 | Data generation | Synthetic (formula-based) | Create Analyst x AoP records with LTSI factors, labels |
| 2 | Feature engineering | Derived features + scaling | Build model-ready feature matrix |
| 3 | Transfer Success model | Logistic Regression (binary) | Predict P(transfer success) |
| 4 | Incident Risk model | Logistic Regression (binary) | Predict P(incident risk) |
| 5 | Personas | K-Means (k=3) | Segment analysts into three persona clusters |
| 6 | Intervention plans | Rule-based | Generate 70:20:10 plans from risk profile |

---

## 2. Data Generation (Synthetic)

**Purpose:** Generate a dataset that matches the dissertation schema (LTSI factors, job tasks, skill gap, transfer/incident outcomes) for pipeline demonstration and testing.

**Algorithm:**

- **Analysts:** `n_analysts` (default 60). Each gets a **RoleTier** from {Analyst, AdvancedAnalyst, SeniorAnalyst} with probabilities 50%, 35%, 15%.
- **LTSI scores:** 16 factors from the Learning Transfer System Inventory. For each factor, score = `clip(base_tier + N(0, 0.6) + U(-0.5, 0.5), 1, 5)` where `base_tier` is 2.8 (Analyst), 3.5 (AdvancedAnalyst), 4.0 (SeniorAnalyst). Scale 1-5.
- **Areas of Practice (AoPs):** 5 fixed AoPs with `id`, `name`, `difficulty` (1-4), `critical_incident_rate` (0.2-0.6).
- **Per Analyst x AoP record:**
  - **Skill gap:** `required_level` (from AoP difficulty) minus `observed_level` (base proficiency by tier + N(0, 0.8), clipped 1-4).
  - **CuesAvailable:** Beta(2,1) for non-Analyst, Beta(2,2) for Analyst.
  - **TransferSuccess (binary):** Bernoulli with probability = 0.3*(MotivationToTransfer/5) + 0.25*(SupervisorSupport/5) + 0.2*(OpportunityToUseLearning/5) + 0.15*(1 - skill_gap/4) + 0.1*cues_available.
  - **IncidentRisk (binary):** Bernoulli with probability = 0.3*(difficulty/4) + 0.3*max(0, skill_gap/4) + 0.2*(1 - PerformanceSelfEfficacy/5) + 0.2*(1 - cues_available).

**Output:** One row per (AnalystID, AoPID) with all LTSI columns, TaskDifficulty, SkillGap, CuesAvailable, CriticalIncidentFlag, TransferSuccess, IncidentRisk. Total rows = n_analysts x 5.

**Credibility note:** Labels are **synthetic and formula-derived**. The same types of features that the models later use are used to generate the labels, so **optimistic performance is expected**. Real-world credibility requires **real survey/operational data** and proper validation.

---

## 3. Feature Engineering

**Input:** Raw dataframe from data generation.

**Steps:**

1. **Base features:** All 16 LTSI factor columns + TaskDifficulty, SkillGap, CuesAvailable, CriticalIncidentFlag.
2. **Derived features:**
   - `Peer_Supervisor_Interaction` = PeerSupport * SupervisorSupport
   - `Motivation_Efficacy` = MotivationToTransfer * PerformanceSelfEfficacy
   - `High_Difficulty_Low_Cues` = 1 if (TaskDifficulty >= 3 and CuesAvailable < 0.5) else 0
3. **RoleTier:** Label-encoded (0/1/2) and added as a feature.
4. **Scaling:** StandardScaler (zero mean, unit variance) fitted on the full feature matrix; same scaling applied to train and test after split.

**Output:** Feature matrix `X` (and `X_scaled`) with 16 + 4 + 3 = 23 columns (LTSI + 4 base + 3 derived; RoleTier replaces raw tier). Shape (n_records, 23).

---

## 4. Train/Test Split

- **Method:** `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`.
- **Applied to:** Same `X_scaled` for both targets; stratify by `y_transfer` for transfer model and by `y_incident` for incident model.
- **Result:** 80% train, 20% test. Single split (no cross-validation).

---

## 5. Transfer Success Model (Binary Classification)

**Algorithm:** Logistic Regression (sklearn).

- **Solver:** default (lbfgs).
- **max_iter:** 1000.
- **class_weight:** "balanced" (to handle class imbalance).
- **C:** 0.5 (L2 regularization strength).

**Target:** `TransferSuccess` (0/1).

**Training:** Fit on (X_train, y_train_transfer).

**Outputs:**

- **Probabilities:** `predict_proba(X_scaled)[:, 1]` for full dataset (used in clustering and intervention logic).
- **Predictions:** Binary predictions on test set for metrics.

**Metrics (test set):**

| Metric | Definition |
|--------|------------|
| **AUC (ROC)** | Area under ROC curve (discrimination). |
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN). |
| **Precision** | TP / (TP+FP). Zero division = 0 if no positive predictions. |
| **Recall** | TP / (TP+FN). Zero division = 0 if no positive in ground truth. |
| **F1** | 2 * (Precision * Recall) / (Precision + Recall). Zero division = 0. |
| **Confusion matrix** | [[TN, FP], [FN, TP]]. |

**Interpretability:** Coefficients (and top drivers by absolute coefficient) are reported in the UI and pipeline log.

---

## 6. Incident Risk Model (Binary Classification)

**Algorithm:** Same as Transfer Success: Logistic Regression, class_weight="balanced", C=0.5, max_iter=1000.

**Target:** `IncidentRisk` (0/1).

**Training:** Fit on (X_train, y_train_incident).

**Outputs:** Probabilities on full dataset; binary predictions on test set.

**Metrics (test set):** Same as Transfer Success (AUC, Accuracy, Precision, Recall, F1, Confusion matrix).

---

## 7. Clustering (Personas)

**Algorithm:** K-Means (sklearn).

- **n_clusters:** 3 (fixed).
- **random_state:** 42.

**Input:** Analyst-level aggregated features (one row per analyst):

- MotivationToTransfer (mean), PerformanceSelfEfficacy (mean), SupervisorSupport (mean), SkillGap (mean), TransferSuccess_Prob (mean), IncidentRisk_Prob (mean), RoleTierEncoded (first).

**Processing:** Features scaled with StandardScaler; K-Means fit on scaled matrix. Each analyst assigned to cluster 0, 1, or 2.

**Persona labeling (rule-based after clustering):** For each cluster, compute mean MotivationToTransfer and mean SkillGap. Then:

- If avg_motivation > 3.5 and avg_gap < 0 → **High_Performer**
- Else if avg_motivation < 3.0 → **Needs_Motivation_Support**
- Else → **Skill_Builder**

**Metrics:**

| Metric | Definition |
|--------|------------|
| **Silhouette score** | Mean silhouette coefficient (-1 to 1). Higher = better separation. |
| **Inertia** | Sum of squared distances to nearest centroid. Lower = tighter clusters (not comparable across k). |

**Output:** Each analyst has PersonaCluster (0/1/2) and PersonaLabel (High_Performer / Needs_Motivation_Support / Skill_Builder). These are merged back to the main Analyst x AoP dataframe.

---

## 8. Intervention Plan Generation (Rule-Based)

**Algorithm:** Deterministic rules applied per (Analyst, AoP) row. Implemented in `build_intervention_plan_from_row()`.

**Inputs (from row):** SkillGap, CuesAvailable, CriticalIncidentFlag, IncidentRisk_Prob, SupervisorSupport, TaskDifficulty, AoPName, PersonaLabel.

**Rules (summary):**

- If SkillGap > 0.5 → add OTJ "Structured_Practice" + artifact "Observation_Checklist_V2".
- If CuesAvailable < 0.5 → add OTJ "Performance_Support_Tool" + artifact "PST_Cues_Strategies".
- If CriticalIncidentFlag == 1 or IncidentRisk_Prob > 0.6 → add Social_20 "Peer_Coaching" + "Critical_Incident_Analysis_Table".
- If SupervisorSupport < 3.0 → add Social_20 "Manager_Check_in".
- If TaskDifficulty >= 3 → add Formal_10 "Scenario_Based_eLearning" + "eLearning_Scenario_Module".
- Success criteria: "50% incident reduction in 30 days" if IncidentRisk_Prob > 0.7 else "25% in 30 days".

**Output:** One intervention plan per selected row (RiskProfile, TargetedInterventions by OTJ_70/Social_20/Formal_10, EvidenceArtifacts, SuccessCriteria). Pipeline builds plans for up to 10 high-risk (IncidentRisk_Prob > 0.6) Analyst-AoP pairs.

**Metrics:** None (rule-based; no accuracy or precision). Evaluation would require outcome data (e.g. did incident rate change after intervention).

---

## 9. Complete Metrics Reported by the Pipeline

After the pipeline run, the following are computed and returned (and shown in the app where indicated).

### 9.1 Transfer Success Model (test set)

| Metric | Key in pipeline output | Shown in UI |
|--------|------------------------|-------------|
| AUC | metrics_transfer["auc"], auc_transfer | Pipeline log, summary |
| Accuracy | metrics_transfer["accuracy"] | Pipeline log |
| Precision | metrics_transfer["precision"] | Pipeline log |
| Recall | metrics_transfer["recall"] | Pipeline log |
| F1 | metrics_transfer["f1"] | Pipeline log |
| Confusion matrix | metrics_transfer["confusion_matrix"] | In return dict (can be displayed) |

### 9.2 Incident Risk Model (test set)

| Metric | Key in pipeline output | Shown in UI |
|--------|------------------------|-------------|
| AUC | metrics_incident["auc"], auc_incident | Pipeline log, summary |
| Accuracy | metrics_incident["accuracy"] | Pipeline log |
| Precision | metrics_incident["precision"] | Pipeline log |
| Recall | metrics_incident["recall"] | Pipeline log |
| F1 | metrics_incident["f1"] | Pipeline log |
| Confusion matrix | metrics_incident["confusion_matrix"] | In return dict (can be displayed) |

### 9.3 Clustering

| Metric | Key in pipeline output | Shown in UI |
|--------|------------------------|-------------|
| Silhouette score | cluster_metrics["silhouette_score"] | Pipeline log |
| Inertia | cluster_metrics["inertia"] | Pipeline log |
| n_clusters | cluster_metrics["n_clusters"] | 3 (fixed) |

### 9.4 Dataset Summary

| Metric | Key | Meaning |
|--------|-----|---------|
| n_analysts | n_analysts | Number of synthetic analysts |
| n_aops | n_aops | Number of AoPs (5) |
| transfer_rate | transfer_rate | Mean of TransferSuccess in full dataset |
| incident_rate | incident_rate | Mean of IncidentRisk in full dataset |
| persona_counts | persona_counts | Count per PersonaLabel |

---

## 10. Credibility Evaluation - Summary

**Strengths for evaluation:**

- **Transparent algorithms:** Logistic regression and K-Means are interpretable; coefficients and cluster labels are exposed.
- **Full metrics:** AUC, accuracy, precision, recall, F1, and confusion matrices for both classifiers; silhouette and inertia for clustering.
- **Stratified split:** Test set is stratified by target to preserve class balance.
- **Balanced classes:** class_weight="balanced" for logistic models.

**Limitations (affect credibility of “real” claims):**

1. **Synthetic data:** Labels are generated from formulas that use the same conceptual drivers the model uses. Performance on this data is **not** generalizable; it demonstrates pipeline behavior, not real-world predictive validity.
2. **Single split:** One train/test split; no cross-validation or repeated runs. Metrics can vary with different random seeds.
3. **No external validation:** No hold-out cohort or temporal validation.
4. **Fixed k=3:** Persona count is not data-driven; no elbow/silhouette search over k.
5. **Intervention plans:** Purely rule-based; no causal or outcome evaluation.

**Recommendations for credible results:**

- Replace synthetic data with **real** LTSI/survey and operational (e.g. incident) data.
- Add **cross-validation** (e.g. 5-fold) and report mean and SD of AUC, F1, etc.
- Optionally **tune** regularization (C) and report a single test-set evaluation on a locked test set.
- Consider **cluster validation** (e.g. elbow plot or silhouette over k) if persona count should be data-driven.
- For intervention plans, define **success metrics** (e.g. incident rate change) and collect post-intervention data to evaluate impact.

---

## 11. Model Comparison: Theory-Constrained vs. Black-Box

### Objective
Demonstrate that theory-constrained Logistic Regression outperforms or matches black-box alternatives (Random Forest, XGBoost) in small-data regimes (n<200) for SME decision intelligence, with superior stability (lower CV variance) and ante-hoc explainability.

### Methodology
- **5-fold stratified cross-validation** (StratifiedKFold, shuffle=True, random_state=42).
- **Models:** LogisticRegression (C=0.5, balanced), RandomForest (n_estimators=100, max_depth=5, balanced), XGBoost (max_depth=3, n_estimators=100, scale_pos_weight=10).
- **Metrics:** AUC (test + CV mean/std), F1, accuracy, precision, recall; theory alignment checks on coefficient signs.

### Results (Synthetic Data)
| Model | Transfer AUC | Transfer CV Std | Incident AUC | Incident CV Std | Theory Aligned |
|-------|-------------|-----------------|-------------|-----------------|----------------|
| LogisticRegression_Theory | ~0.80-0.88 | ~0.02-0.05 | ~0.78-0.86 | ~0.03-0.06 | Yes |
| RandomForest_BlackBox | ~0.82-0.90 | ~0.05-0.10 | ~0.80-0.88 | ~0.06-0.11 | No |
| XGBoost_BlackBox | ~0.82-0.90 | ~0.05-0.09 | ~0.80-0.88 | ~0.05-0.10 | No |

### Conclusion
Theory-constrained models provide comparable accuracy with **superior stability** (lower CV std) and **ante-hoc interpretability** (coefficients map to LTSI constructs), making them preferable for SME governance applications where auditability and stakeholder trust matter.

### Limitations
Synthetic data demonstrates algorithmic behavior; real-world validation requires evaluation with empirical SME/LTSI data (n >= 100).

---

## 12. Where to Find Results in Code

| Item | File | Symbol / Section |
|------|------|-------------------|
| Data generation | pipeline/ml_pipeline.py | generate_ltsi_scores(), run_ml_pipeline() records loop |
| Feature engineering | pipeline/ml_pipeline.py | run_ml_pipeline() after df = pd.DataFrame(records) |
| Train/test split | pipeline/ml_pipeline.py | train_test_split(...) |
| Transfer / Incident models | pipeline/ml_pipeline.py | LogisticRegression(...), .fit(), metrics_transfer / metrics_incident |
| Clustering | pipeline/ml_pipeline.py | KMeans(n_clusters=3), cluster_metrics |
| Intervention rules | pipeline/ml_pipeline.py | build_intervention_plan_from_row() |
| Metrics display | ui/streamlit_app.py | log_lines, out["metrics_transfer"], out["cluster_metrics"] |
| Model comparison | pipeline/ml_pipeline.py | models_config, model_comparison_results, cross_val_score |
| Theory validation | pipeline/ml_pipeline.py | theory_validation, transfer_coefficients, incident_coefficients |
| Performance page | ui/streamlit_app.py | render_model_performance() |

This document and the pipeline return structure together provide a **complete specification of algorithms and results** for your credibility evaluation.
