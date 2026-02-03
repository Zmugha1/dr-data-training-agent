"""
Human-in-the-Loop Decision Intelligence for Training Transfer Optimization.
Main Streamlit interface: Dashboard, Decision Queue, AoP Ingestion, Intervention Planner, Analytics.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from core.schema import (
    HumanDecision,
    InterventionPlan,
    JobTask,
)
from core.state_manager import StateManager
from human_loop.decision_interface import DecisionManager
from agents.agent_01_context_ingestion import ContextIngestionAgent
from agents.agent_08_intervention_planner import InterventionPlannerAgent


# --- Config ---
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_ROOT = ROOT / "data"
PENDING_DIR = DATA_ROOT / "pending_decisions"


def load_config() -> dict:
    """Load config from YAML."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_decision_manager() -> DecisionManager:
    """Session-scoped DecisionManager."""
    if "decision_manager" not in st.session_state:
        st.session_state.decision_manager = DecisionManager(pending_dir=PENDING_DIR)
    return st.session_state.decision_manager


def get_state_manager() -> StateManager:
    """Session-scoped StateManager."""
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = StateManager(data_root=DATA_ROOT)
    return st.session_state.state_manager


def get_ingestion_agent() -> ContextIngestionAgent:
    if "ingestion_agent" not in st.session_state:
        st.session_state.ingestion_agent = ContextIngestionAgent(get_decision_manager())
    return st.session_state.ingestion_agent


def get_planner_agent() -> InterventionPlannerAgent:
    if "planner_agent" not in st.session_state:
        st.session_state.planner_agent = InterventionPlannerAgent(get_decision_manager())
    return st.session_state.planner_agent


# --- Sidebar ---
def render_sidebar(config: dict) -> None:
    ui = config.get("ui", {})
    expert_name = ui.get("expert_name", "Dr. Zubia Mughal")
    expert_role = ui.get("expert_role", "Human Expert (Governance)")
    dm = get_decision_manager()
    pending = dm.get_pending_decisions()

    st.sidebar.title("Human-in-the-Loop")
    st.sidebar.markdown("**70:20:10 Training Transfer**")
    st.sidebar.divider()
    st.sidebar.markdown(f"**Human expert profile**  \n{expert_name}  \n*{expert_role}*")
    st.sidebar.divider()
    st.sidebar.metric("Pending decisions", len(pending))
    st.sidebar.metric("Governance mode", config.get("governance", {}).get("governance_mode", "human_in_the_loop"))
    st.sidebar.divider()
    st.sidebar.caption(f"Status at {datetime.utcnow().strftime('%H:%M:%S')} UTC")


# --- Tabs ---
def tab_dashboard(config: dict) -> None:
    sm = get_state_manager()
    dm = get_decision_manager()
    pending = dm.get_pending_decisions()
    plans = sm.get_intervention_plans()
    tasks = sm.get_job_tasks()

    # 70:20:10 sunburst data
    labels = ["70:20:10", "OTJ 70%", "Social 20%", "Formal 10%"]
    parents = ["", "70:20:10", "70:20:10", "70:20:10"]
    values = [100, 70, 20, 10]
    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        )
    )
    fig.update_layout(title="70:20:10 Learning Model", height=400, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # High-risk / pending table
    st.subheader("High-risk & pending")
    if pending:
        rows = []
        for d in pending[:20]:
            risk = d.proposal_payload.get("risk_profile") or {}
            incident = risk.get("incident_risk", 0)
            skill = risk.get("skill_gap", 0)
            rows.append({
                "Decision ID": d.decision_id[:8] + "...",
                "Type": d.proposal_type,
                "Incident risk": f"{incident:.2f}" if isinstance(incident, (int, float)) else "-",
                "Skill gap": f"{skill:.2f}" if isinstance(skill, (int, float)) else "-",
                "Created": d.created_at.strftime("%Y-%m-%d %H:%M") if d.created_at else "-",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No pending decisions. Use **Decision Queue** or **Intervention Planner** to create proposals.")


def tab_decision_queue() -> None:
    dm = get_decision_manager()
    sm = get_state_manager()
    config = load_config()
    ui = config.get("ui", {})
    expert_name = ui.get("expert_name", "Dr. Zubia Mughal")

    st.subheader("Decision Queue")
    pending = dm.get_pending_decisions()
    if not pending:
        st.info("No pending decisions.")
        return

    for d in pending:
        with st.expander(f"**{d.proposal_type}** — {d.decision_id[:8]}... — {d.created_at.strftime('%Y-%m-%d %H:%M') if d.created_at else 'N/A'}", expanded=True):
            st.json(d.proposal_payload)
            rationale = st.text_area("Rationale (optional)", key=f"rationale_{d.decision_id}", placeholder="Reason for approve/reject/modify")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Approve", key=f"approve_{d.decision_id}"):
                    dm.make_decision(d.decision_id, "approved", human_actor=expert_name, human_rationale=rationale)
                    if d.proposal_type == "job_task":
                        try:
                            task = JobTask.model_validate({k: v for k, v in d.proposal_payload.items() if k != "_confidence"})
                            sm.add_job_task(task)
                        except Exception:
                            pass
                    elif d.proposal_type == "intervention_plan":
                        try:
                            plan = InterventionPlan.model_validate(d.proposal_payload)
                            sm.add_intervention_plan(plan)
                        except Exception:
                            pass
                    st.success("Approved.")
                    st.rerun()
            with col2:
                if st.button("Reject", key=f"reject_{d.decision_id}"):
                    dm.make_decision(d.decision_id, "rejected", human_actor=expert_name, human_rationale=rationale)
                    st.info("Rejected.")
                    st.rerun()
            with col3:
                if st.button("Modify", key=f"modify_{d.decision_id}"):
                    modified = d.proposal_payload.copy()
                    dm.make_decision(d.decision_id, "modified", human_actor=expert_name, human_rationale=rationale, modified_payload=modified)
                    st.warning("Marked as modified. Update payload in audit if needed.")
                    st.rerun()


def tab_aop_ingestion() -> None:
    agent = get_ingestion_agent()
    st.subheader("AoP Ingestion (Verb–Task–Product)")
    raw = st.text_area("SME design document or raw text", height=200, placeholder="e.g.: Analyze customer feedback → produce weekly report")
    if raw.strip():
        tasks = agent.parse_text(raw.strip())
        confidence = [agent.confidence_score(t) for t in tasks]
        for i, (t, c) in enumerate(zip(tasks, confidence)):
            st.markdown(f"**Parsed {i+1}:** `{t.to_verb_task_product()}` — confidence: `{c:.2f}`")
        if st.button("Submit to Decision Queue (human validation)"):
            results = agent.ingest_and_submit(raw.strip(), submit_all=True)
            for _task, _c, did in results:
                st.success(f"Queued for human approval. Decision ID: {did}")
            st.rerun()
    else:
        st.caption("Enter text and click submit to queue parsed JobTasks for human approval.")


def tab_intervention_planner() -> None:
    planner = get_planner_agent()
    sm = get_state_manager()
    config = load_config()
    ui = config.get("ui", {})
    expert_name = ui.get("expert_name", "Dr. Zubia Mughal")

    st.subheader("Intervention Planner (70:20:10)")
    analyst_id = st.selectbox("Analyst", ["analyst_1", "analyst_2", "analyst_3", expert_name], key="analyst_select")
    incident_risk = st.slider("Incident risk", 0.0, 1.0, 0.5, 0.1)
    skill_gap = st.slider("Skill gap", 0.0, 1.0, 0.5, 0.1)
    difficulty = st.selectbox("Difficulty", [1, 2, 3, 4], index=1, format_func=lambda x: f"Level {x}")
    risk_profile = {"incident_risk": incident_risk, "skill_gap": skill_gap}

    if st.button("Generate Plan"):
        plan = planner.generate_plan(
            analyst_id=analyst_id,
            risk_profile=risk_profile,
            difficulty=difficulty,
        )
        st.session_state["last_plan"] = plan

    if "last_plan" in st.session_state:
        plan = st.session_state["last_plan"]
        st.markdown("**70:20:10 breakdown**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**OTJ 70%**")
            for a in plan.otj_70_activities:
                st.write(f"- {a}")
        with col2:
            st.markdown("**Social 20%**")
            for a in plan.social_20_activities:
                st.write(f"- {a}")
        with col3:
            st.markdown("**Formal 10%**")
            for a in plan.formal_10_activities:
                st.write(f"- {a}")
        if st.button("Submit plan to Decision Queue"):
            dm = get_decision_manager()
            decision = dm.propose_decision(
                proposal_type="intervention_plan",
                proposal_payload=plan.model_dump(mode="json"),
            )
            if "last_plan" in st.session_state:
                del st.session_state["last_plan"]
            st.success(f"Plan submitted for human approval. Decision ID: {decision.decision_id}")
            st.rerun()


def tab_analytics() -> None:
    sm = get_state_manager()
    dm = get_decision_manager()
    pending = dm.get_pending_decisions()
    plans = sm.get_intervention_plans()
    tasks = sm.get_job_tasks()

    st.subheader("Analytics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pending decisions", len(pending))
    c2.metric("Intervention plans", len(plans))
    c3.metric("Job tasks", len(tasks))
    if plans:
        st.markdown("**Plans by risk**")
        rows = []
        for p in plans:
            rp = p.risk_profile or {}
            rows.append({
                "Plan ID": p.plan_id[:8] + "...",
                "Incident risk": rp.get("incident_risk", 0),
                "Skill gap": rp.get("skill_gap", 0),
                "OTJ count": len(p.otj_70_activities),
                "Social count": len(p.social_20_activities),
                "Formal count": len(p.formal_10_activities),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Human-in-the-Loop Decision Intelligence",
        page_icon="⚖️",
        layout="wide",
    )
    config = load_config()
    render_sidebar(config)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard",
        "⚖️ Decision Queue",
        "📥 AoP Ingestion",
        "🎯 Intervention Planner",
        "📈 Analytics",
    ])
    with tab1:
        tab_dashboard(config)
    with tab2:
        tab_decision_queue()
    with tab3:
        tab_aop_ingestion()
    with tab4:
        tab_intervention_planner()
    with tab5:
        tab_analytics()


if __name__ == "__main__":
    main()
