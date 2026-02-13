"""
Decision Intelligence Training-Transfer Agent: ML Pipeline
Implements: Feature Eng → Logistic Reg → Clustering → 70:20:10
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import warnings
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.metrics import silhouette_score
from sklearn.base import clone
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

LTSI_FACTORS = [
    "LearnerReadiness", "MotivationToTransfer", "PositivePersonalOutcomes",
    "NegativePersonalOutcomes", "PersonalCapacityForTransfer", "PeerSupport",
    "SupervisorSupport", "SupervisorSanctions", "PerceivedContentValidity",
    "TransferDesign", "OpportunityToUseLearning", "PerformanceSelfEfficacy",
    "PerformanceOutcomeExpectations", "PerceivedOrganizationalSupport",
    "ResistanceToChange", "TaskCues",
]

DEFAULT_AOPS = [
    {"id": "AOP_01", "name": "Information_Gathering", "difficulty": 2, "critical_incident_rate": 0.3},
    {"id": "AOP_02", "name": "Diagnosis_Triage", "difficulty": 3, "critical_incident_rate": 0.5},
    {"id": "AOP_03", "name": "Routing_Handoff", "difficulty": 2, "critical_incident_rate": 0.2},
    {"id": "AOP_04", "name": "Resolution_Documentation", "difficulty": 3, "critical_incident_rate": 0.4},
    {"id": "AOP_05", "name": "Customer_Communication", "difficulty": 4, "critical_incident_rate": 0.6},
]


def generate_ltsi_scores(tier: str) -> dict:
    base = {"Analyst": 2.8, "AdvancedAnalyst": 3.5, "SeniorAnalyst": 4.0}[tier]
    scores = {}
    for factor in LTSI_FACTORS:
        noise = np.random.normal(0, 0.6)
        score = np.clip(base + noise + np.random.uniform(-0.5, 0.5), 1, 5)
        scores[factor] = round(float(score), 2)
    return scores


def build_intervention_plan_from_row(row: pd.Series, analyst_id: str, aop_id: str) -> dict:
    """
    Build a single targeted intervention plan from one Analyst x AoP row (e.g. from pipeline df).
    Used by run_ml_pipeline and by the UI when user selects analyst + AoP.
    """
    plan = {
        "AnalystID": analyst_id,
        "AoPID": aop_id,
        "GeneratedTimestamp": datetime.now().isoformat(),
        "RiskProfile": {
            "IncidentRisk_Prob": round(float(row["IncidentRisk_Prob"]), 3),
            "TransferSuccess_Prob": round(float(row["TransferSuccess_Prob"]), 3),
            "Persona": row["PersonaLabel"],
            "SkillGap": float(row["SkillGap"]),
        },
        "TargetedInterventions": {"OTJ_70": [], "Social_20": [], "Formal_10": []},
        "EvidenceArtifacts": [],
        "SuccessCriteria": {},
    }
    aop_name = row["AoPName"]
    if row["SkillGap"] > 0.5:
        plan["TargetedInterventions"]["OTJ_70"].append({
            "Type": "Structured_Practice",
            "Task": f"Practice {aop_name} with supervisor checklist",
            "Duration": "2 weeks",
        })
        plan["EvidenceArtifacts"].append("Observation_Checklist_V2")
    if row["CuesAvailable"] < 0.5:
        plan["TargetedInterventions"]["OTJ_70"].append({
            "Type": "Performance_Support_Tool",
            "Task": "Deploy job aid with cues/strategies",
        })
        plan["EvidenceArtifacts"].append("PST_Cues_Strategies")
    if row["CriticalIncidentFlag"] == 1 or row["IncidentRisk_Prob"] > 0.6:
        plan["TargetedInterventions"]["Social_20"].append({
            "Type": "Peer_Coaching",
            "Topic": f"Critical incident review: {aop_name}",
        })
        plan["EvidenceArtifacts"].append("Critical_Incident_Analysis_Table")
    if row["SupervisorSupport"] < 3.0:
        plan["TargetedInterventions"]["Social_20"].append({
            "Type": "Manager_Check_in",
            "Frequency": "Weekly",
        })
    if row["TaskDifficulty"] >= 3:
        plan["TargetedInterventions"]["Formal_10"].append({
            "Type": "Scenario_Based_eLearning",
            "Topic": f"{aop_name} - Difficult task mastery",
            "Duration": "30 mins",
        })
        plan["EvidenceArtifacts"].append("eLearning_Scenario_Module")
    plan["SuccessCriteria"] = {
        "Incident_Reduction": "50% in 30 days" if row["IncidentRisk_Prob"] > 0.7 else "25% in 30 days",
        "Measurement_Window": "30 days",
    }
    return plan


def run_ml_pipeline(n_analysts: int = 60) -> dict:
    """
    Run full ML pipeline: synthetic data → feature eng → models → clustering → intervention plans.
    Returns dict with df, analyst_features, intervention_plans, intervention_df, model metrics, etc.
    """
    analyst_ids = [f"ANALYST_{i:03d}" for i in range(n_analysts)]
    role_tiers = np.random.choice(
        ["Analyst", "AdvancedAnalyst", "SeniorAnalyst"],
        n_analysts,
        p=[0.5, 0.35, 0.15],
    )
    ltsi_data = [generate_ltsi_scores(tier) for tier in role_tiers]
    aops = list(DEFAULT_AOPS)

    records = []
    for i, analyst in enumerate(analyst_ids):
        tier = role_tiers[i]
        ltsi = ltsi_data[i]
        base_proficiency = {"Analyst": 1.5, "AdvancedAnalyst": 2.5, "SeniorAnalyst": 3.5}[tier]
        for aop in aops:
            required_level = aop["difficulty"]
            observed_level = np.clip(base_proficiency + np.random.normal(0, 0.8), 1, 4)
            skill_gap = required_level - observed_level
            incident_exposure = 1 if np.random.random() < aop["critical_incident_rate"] else 0
            cues_available = np.random.beta(2, 1) if tier != "Analyst" else np.random.beta(2, 2)
            transfer_prob = (
                0.3 * (ltsi["MotivationToTransfer"] / 5)
                + 0.25 * (ltsi["SupervisorSupport"] / 5)
                + 0.2 * (ltsi["OpportunityToUseLearning"] / 5)
                + 0.15 * (1 - skill_gap / 4)
                + 0.1 * cues_available
            )
            transfer_success = 1 if np.random.random() < transfer_prob else 0
            risk_prob = (
                0.3 * (aop["difficulty"] / 4)
                + 0.3 * max(0, skill_gap / 4)
                + 0.2 * (1 - ltsi["PerformanceSelfEfficacy"] / 5)
                + 0.2 * (1 - cues_available)
            )
            incident_risk = 1 if np.random.random() < risk_prob else 0
            records.append({
                "AnalystID": analyst,
                "RoleTier": tier,
                "AoPID": aop["id"],
                "AoPName": aop["name"],
                "TaskDifficulty": aop["difficulty"],
                "CriticalIncidentFlag": incident_exposure,
                "SkillGap": round(float(skill_gap), 2),
                "ObservedProficiency": round(float(observed_level), 2),
                "CuesAvailable": round(float(cues_available), 2),
                "TransferSuccess": transfer_success,
                "IncidentRisk": incident_risk,
                **ltsi,
            })

    df = pd.DataFrame(records)

    # Feature engineering
    feature_cols = LTSI_FACTORS + ["TaskDifficulty", "SkillGap", "CuesAvailable", "CriticalIncidentFlag"]
    X = df[feature_cols].copy()
    y_transfer = df["TransferSuccess"]
    y_incident = df["IncidentRisk"]
    X["Peer_Supervisor_Interaction"] = X["PeerSupport"] * X["SupervisorSupport"]
    X["Motivation_Efficacy"] = X["MotivationToTransfer"] * X["PerformanceSelfEfficacy"]
    X["High_Difficulty_Low_Cues"] = ((X["TaskDifficulty"] >= 3) & (X["CuesAvailable"] < 0.5)).astype(int)
    le_tier = LabelEncoder()
    df["RoleTierEncoded"] = le_tier.fit_transform(df["RoleTier"])
    X["RoleTier"] = df["RoleTierEncoded"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train_transfer, y_test_transfer = train_test_split(
        X_scaled, y_transfer, test_size=0.2, random_state=42, stratify=y_transfer
    )
    _, _, y_train_incident, y_test_incident = train_test_split(
        X_scaled, y_incident, test_size=0.2, random_state=42, stratify=y_incident
    )

    # ============================================================
    # THEORY-CONSTRAINED VS BLACK-BOX MODEL COMPARISON
    # ============================================================
    # Research Question: Do theory-constrained (interpretable) models outperform
    # black-box alternatives in small-data regimes (n<200) for SME decision intelligence?

    models_config = {
        "LogisticRegression_Theory": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5),
            "is_theory_based": True,
            "description": "LTSI-theory guided, ante-hoc explainable",
        },
        "RandomForest_BlackBox": {
            "model": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, max_depth=5),
            "is_theory_based": False,
            "description": "Ensemble method, post-hoc explanation required",
        },
        "XGBoost_BlackBox": {
            "model": xgb.XGBClassifier(
                scale_pos_weight=10, max_depth=3, random_state=42, eval_metric="logloss", n_estimators=100
            ),
            "is_theory_based": False,
            "description": "Gradient boosting, post-hoc explanation required",
        },
    }

    model_comparison_results: Dict[str, Any] = {}
    cv_folds = 5
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Train and evaluate all models for Transfer Success
    for model_name, config in models_config.items():
        model = clone(config["model"])
        cv_scores_transfer = cross_val_score(
            model, X_scaled, y_transfer, cv=cv_splitter, scoring="roc_auc"
        )
        model.fit(X_train, y_train_transfer)
        y_pred_transfer = model.predict(X_test)
        y_prob_transfer = model.predict_proba(X_test)[:, 1]
        model_comparison_results[model_name] = {
            "transfer": {
                "auc": float(roc_auc_score(y_test_transfer, y_prob_transfer)),
                "accuracy": float(accuracy_score(y_test_transfer, y_pred_transfer)),
                "precision": float(precision_score(y_test_transfer, y_pred_transfer, zero_division=0)),
                "recall": float(recall_score(y_test_transfer, y_pred_transfer, zero_division=0)),
                "f1": float(f1_score(y_test_transfer, y_pred_transfer, zero_division=0)),
                "cv_auc_mean": float(cv_scores_transfer.mean()),
                "cv_auc_std": float(cv_scores_transfer.std()),
                "cv_scores": cv_scores_transfer.tolist(),
            },
            "is_theory_based": config["is_theory_based"],
            "description": config["description"],
        }

    # Repeat for Incident Risk (fresh clone per model)
    for model_name, config in models_config.items():
        model = clone(config["model"])
        cv_scores_incident = cross_val_score(
            model, X_scaled, y_incident, cv=cv_splitter, scoring="roc_auc"
        )
        model.fit(X_train, y_train_incident)
        y_pred_incident = model.predict(X_test)
        y_prob_incident = model.predict_proba(X_test)[:, 1]
        model_comparison_results[model_name]["incident"] = {
            "auc": float(roc_auc_score(y_test_incident, y_prob_incident)),
            "accuracy": float(accuracy_score(y_test_incident, y_pred_incident)),
            "precision": float(precision_score(y_test_incident, y_pred_incident, zero_division=0)),
            "recall": float(recall_score(y_test_incident, y_pred_incident, zero_division=0)),
            "f1": float(f1_score(y_test_incident, y_pred_incident, zero_division=0)),
            "cv_auc_mean": float(cv_scores_incident.mean()),
            "cv_auc_std": float(cv_scores_incident.std()),
            "cv_scores": cv_scores_incident.tolist(),
        }

    # Select theory-based model for final use (governance requirement)
    final_model_transfer = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
    final_model_transfer.fit(X_train, y_train_transfer)
    final_model_incident = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
    final_model_incident.fit(X_train, y_train_incident)

    df["TransferSuccess_Prob"] = final_model_transfer.predict_proba(X_scaled)[:, 1]
    df["IncidentRisk_Prob"] = final_model_incident.predict_proba(X_scaled)[:, 1]

    y_pred_transfer_final = final_model_transfer.predict(X_test)
    y_pred_incident_final = final_model_incident.predict(X_test)
    metrics_transfer = {
        "auc": float(roc_auc_score(y_test_transfer, final_model_transfer.predict_proba(X_test)[:, 1])),
        "accuracy": float(accuracy_score(y_test_transfer, y_pred_transfer_final)),
        "precision": float(precision_score(y_test_transfer, y_pred_transfer_final, zero_division=0)),
        "recall": float(recall_score(y_test_transfer, y_pred_transfer_final, zero_division=0)),
        "f1": float(f1_score(y_test_transfer, y_pred_transfer_final, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test_transfer, y_pred_transfer_final).tolist(),
    }
    metrics_incident = {
        "auc": float(roc_auc_score(y_test_incident, final_model_incident.predict_proba(X_test)[:, 1])),
        "accuracy": float(accuracy_score(y_test_incident, y_pred_incident_final)),
        "precision": float(precision_score(y_test_incident, y_pred_incident_final, zero_division=0)),
        "recall": float(recall_score(y_test_incident, y_pred_incident_final, zero_division=0)),
        "f1": float(f1_score(y_test_incident, y_pred_incident_final, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test_incident, y_pred_incident_final).tolist(),
    }

    # Ante-hoc explainability: coefficients
    feature_names = X.columns.tolist()
    transfer_coefficients = dict(zip(feature_names, final_model_transfer.coef_[0]))
    incident_coefficients = dict(zip(feature_names, final_model_incident.coef_[0]))

    # Theory alignment validation (LTSI constructs should behave predictably)
    theory_validation = {
        "transfer": {
            "MotivationToTransfer_positive": transfer_coefficients.get("MotivationToTransfer", 0) > 0,
            "PerformanceSelfEfficacy_positive": transfer_coefficients.get("PerformanceSelfEfficacy", 0) > 0,
            "SupervisorSupport_positive": transfer_coefficients.get("SupervisorSupport", 0) > 0,
            "SkillGap_negative": transfer_coefficients.get("SkillGap", 0) < 0,
            "TaskDifficulty_negative": transfer_coefficients.get("TaskDifficulty", 0) < 0,
        },
        "incident": {
            "TaskDifficulty_positive": incident_coefficients.get("TaskDifficulty", 0) > 0,
            "SkillGap_positive": incident_coefficients.get("SkillGap", 0) > 0,
            "PerformanceSelfEfficacy_negative": incident_coefficients.get("PerformanceSelfEfficacy", 0) < 0,
            "CuesAvailable_negative": incident_coefficients.get("CuesAvailable", 0) < 0,
        },
    }

    # Clustering / personas
    analyst_features = df.groupby("AnalystID").agg({
        "MotivationToTransfer": "mean",
        "PerformanceSelfEfficacy": "mean",
        "SupervisorSupport": "mean",
        "SkillGap": "mean",
        "TransferSuccess_Prob": "mean",
        "IncidentRisk_Prob": "mean",
        "RoleTierEncoded": "first",
    }).reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42)
    analyst_features["PersonaCluster"] = kmeans.fit_predict(
        scaler.fit_transform(analyst_features.drop(["AnalystID"], axis=1))
    )
    cluster_labels = {}
    for cluster in range(3):
        subset = analyst_features[analyst_features["PersonaCluster"] == cluster]
        avg_motivation = subset["MotivationToTransfer"].mean()
        avg_gap = subset["SkillGap"].mean()
        if avg_motivation > 3.5 and avg_gap < 0:
            label = "High_Performer"
        elif avg_motivation < 3.0:
            label = "Needs_Motivation_Support"
        else:
            label = "Skill_Builder"
        cluster_labels[cluster] = label
    analyst_features["PersonaLabel"] = analyst_features["PersonaCluster"].map(cluster_labels)
    df = df.merge(analyst_features[["AnalystID", "PersonaCluster", "PersonaLabel"]], on="AnalystID")

    # Clustering metrics (analyst-level scaled features)
    X_analyst = analyst_features.drop(["AnalystID", "PersonaCluster", "PersonaLabel"], axis=1)
    X_analyst_scaled = scaler.fit_transform(X_analyst)
    cluster_metrics = {
        "silhouette_score": float(silhouette_score(X_analyst_scaled, analyst_features["PersonaCluster"])),
        "inertia": float(kmeans.inertia_),
        "n_clusters": 3,
    }

    # Intervention plan generator (uses shared builder so UI can build plan for any selected row)
    high_risk_cases = df[df["IncidentRisk_Prob"] > 0.6][["AnalystID", "AoPID"]].drop_duplicates().head(10)
    intervention_plans = []
    for _, r in high_risk_cases.iterrows():
        row = df[(df["AnalystID"] == r["AnalystID"]) & (df["AoPID"] == r["AoPID"])].iloc[0]
        intervention_plans.append(build_intervention_plan_from_row(row, r["AnalystID"], r["AoPID"]))

    intervention_rows = []
    for plan in intervention_plans:
        intervention_rows.append({
            "AnalystID": plan["AnalystID"],
            "AoPID": plan["AoPID"],
            "Persona": plan["RiskProfile"]["Persona"],
            "IncidentRisk": plan["RiskProfile"]["IncidentRisk_Prob"],
            "SkillGap": plan["RiskProfile"]["SkillGap"],
            "OTJ_Activities": len(plan["TargetedInterventions"]["OTJ_70"]),
            "Social_Activities": len(plan["TargetedInterventions"]["Social_20"]),
            "Formal_Activities": len(plan["TargetedInterventions"]["Formal_10"]),
            "Artifacts": ", ".join(plan["EvidenceArtifacts"]),
            "KPI_Target": plan["SuccessCriteria"].get("Incident_Reduction", ""),
        })
    intervention_df = pd.DataFrame(intervention_rows)

    # Flatten interventions for pie chart
    intervention_nodes = []
    for plan in intervention_plans:
        for category, items in plan["TargetedInterventions"].items():
            for item in items:
                intervention_nodes.append({
                    "Category": category,
                    "Type": item.get("Type", "General"),
                })
    intervention_df_full = pd.DataFrame(intervention_nodes)

    return {
        "df": df,
        "analyst_features": analyst_features,
        "intervention_plans": intervention_plans,
        "intervention_df": intervention_df,
        "intervention_df_full": intervention_df_full,
        "model_transfer": final_model_transfer,
        "model_incident": final_model_incident,
        "X": X,
        "X_test": X_test,
        "y_test_transfer": y_test_transfer,
        "y_test_incident": y_test_incident,
        "auc_transfer": metrics_transfer["auc"],
        "auc_incident": metrics_incident["auc"],
        "metrics_transfer": metrics_transfer,
        "metrics_incident": metrics_incident,
        "cluster_metrics": cluster_metrics,
        "n_analysts": n_analysts,
        "n_aops": len(aops),
        "aops": aops,
        "persona_counts": analyst_features["PersonaLabel"].value_counts(),
        "transfer_rate": float(df["TransferSuccess"].mean()),
        "incident_rate": float(df["IncidentRisk"].mean()),
        "model_comparison": model_comparison_results,
        "theory_model_selected": True,
        "theory_validation": theory_validation,
        "transfer_coefficients": transfer_coefficients,
        "incident_coefficients": incident_coefficients,
        "cv_folds": cv_folds,
        "features_used": feature_names,
    }
