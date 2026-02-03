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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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

    # Models
    model_transfer = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
    model_transfer.fit(X_train, y_train_transfer)
    model_incident = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
    model_incident.fit(X_train, y_train_incident)

    df["TransferSuccess_Prob"] = model_transfer.predict_proba(X_scaled)[:, 1]
    df["IncidentRisk_Prob"] = model_incident.predict_proba(X_scaled)[:, 1]

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

    # Intervention plan generator
    def generate_plan(analyst_id: str, aop_id: str) -> dict:
        row = df[(df["AnalystID"] == analyst_id) & (df["AoPID"] == aop_id)].iloc[0]
        plan = {
            "AnalystID": analyst_id,
            "AoPID": aop_id,
            "GeneratedTimestamp": datetime.now().isoformat(),
            "RiskProfile": {
                "IncidentRisk_Prob": round(float(row["IncidentRisk_Prob"]), 3),
                "TransferSuccess_Prob": round(float(row["TransferSuccess_Prob"]), 3),
                "Persona": row["PersonaLabel"],
                "SkillGap": row["SkillGap"],
            },
            "TargetedInterventions": {"OTJ_70": [], "Social_20": [], "Formal_10": []},
            "EvidenceArtifacts": [],
            "SuccessCriteria": {},
        }
        if row["SkillGap"] > 0.5:
            plan["TargetedInterventions"]["OTJ_70"].append({
                "Type": "Structured_Practice",
                "Task": f"Practice {row['AoPName']} with supervisor checklist",
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
                "Topic": f"Critical incident review: {row['AoPName']}",
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
                "Topic": f"{row['AoPName']} - Difficult task mastery",
                "Duration": "30 mins",
            })
            plan["EvidenceArtifacts"].append("eLearning_Scenario_Module")
        plan["SuccessCriteria"] = {
            "Incident_Reduction": "50% in 30 days" if row["IncidentRisk_Prob"] > 0.7 else "25% in 30 days",
            "Measurement_Window": "30 days",
        }
        return plan

    high_risk_cases = df[df["IncidentRisk_Prob"] > 0.6][["AnalystID", "AoPID"]].drop_duplicates().head(10)
    intervention_plans = [generate_plan(r["AnalystID"], r["AoPID"]) for _, r in high_risk_cases.iterrows()]

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

    auc_transfer = float(roc_auc_score(y_test_transfer, model_transfer.predict_proba(X_test)[:, 1]))
    auc_incident = float(roc_auc_score(y_test_incident, model_incident.predict_proba(X_test)[:, 1]))

    return {
        "df": df,
        "analyst_features": analyst_features,
        "intervention_plans": intervention_plans,
        "intervention_df": intervention_df,
        "intervention_df_full": intervention_df_full,
        "model_transfer": model_transfer,
        "model_incident": model_incident,
        "X": X,
        "X_test": X_test,
        "y_test_transfer": y_test_transfer,
        "y_test_incident": y_test_incident,
        "auc_transfer": auc_transfer,
        "auc_incident": auc_incident,
        "n_analysts": n_analysts,
        "n_aops": len(aops),
        "aops": aops,
        "persona_counts": analyst_features["PersonaLabel"].value_counts(),
        "transfer_rate": float(df["TransferSuccess"].mean()),
        "incident_rate": float(df["IncidentRisk"].mean()),
    }
