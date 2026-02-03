"""
Human-in-the-Loop Decision Intelligence for Training Transfer Optimization.
Main Streamlit interface: Dashboard, Decision Queue, AoP Ingestion, Intervention Planner, Analytics.
"""
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH (Streamlit Cloud / headless)
ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT.parent):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import yaml
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from core.schema import (
    DecisionStatus,
    InterventionCategory,
    InterventionPlan,
    JobTask,
)


def _get_interventions(interventions: dict, category_value: str) -> list:
    """Get intervention list by category; works with enum or string keys (e.g. after JSON)."""
    if not interventions:
        return []
    try:
        cat = InterventionCategory(category_value)
        return interventions.get(cat, interventions.get(category_value, []))
    except (ValueError, TypeError):
        return interventions.get(category_value, [])
from core.state_manager import StateManager
from human_loop.decision_interface import DecisionManager
from agents.agent_01_context_ingestion import ContextIngestionAgent
from agents.agent_08_intervention_planner import InterventionPlannerAgent
from pipeline.ml_pipeline import run_ml_pipeline


# --- Config ---
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_ROOT = ROOT / "data"
PENDING_DIR = DATA_ROOT / "pending_decisions"
SEED_DATA_PATH = DATA_ROOT / "seed_data.json"


def load_seed_data() -> dict:
    """Load analysts and AOPs from seed_data.json."""
    if not SEED_DATA_PATH.exists():
        return {"analysts": [], "aops": []}
    import json
    with open(SEED_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_seed_data() -> dict:
    """Session-cached seed data."""
    if "seed_data" not in st.session_state:
        st.session_state.seed_data = load_seed_data()
    return st.session_state.seed_data


def load_config() -> dict:
    """Load config from YAML."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_decision_manager() -> DecisionManager:
    """Session-scoped DecisionManager."""
    if "decision_manager" not in st.session_state:
        st.session_state.decision_manager = DecisionManager(pending_dir=str(PENDING_DIR))
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
    seed = get_seed_data()
    st.sidebar.divider()
    st.sidebar.metric("Analysts (seed)", len(seed.get("analysts", [])))
    st.sidebar.metric("AOPs (seed)", len(seed.get("aops", [])))
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
            payload = d.get("proposed_data") or {}
            risk = payload.get("risk_profile") or {}
            incident = risk.get("incident_risk", 0)
            skill = risk.get("skill_gap", 0)
            ts = d.get("timestamp_proposed", "")
            rows.append({
                "Decision ID": (d.get("decision_id") or "")[:8] + "...",
                "Type": d.get("decision_type", ""),
                "Incident risk": f"{incident:.2f}" if isinstance(incident, (int, float)) else "-",
                "Skill gap": f"{skill:.2f}" if isinstance(skill, (int, float)) else "-",
                "Created": str(ts)[:16] if ts else "-",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No pending decisions. Use **Decision Queue** or **Intervention Planner** to create proposals.")

    # Seed data: Analysts & AOPs
    seed = get_seed_data()
    analysts = seed.get("analysts", [])
    aops = seed.get("aops", [])
    if analysts or aops:
        st.subheader("Seed data")
        col_a, col_b = st.columns(2)
        with col_a:
            if analysts:
                st.markdown("**Analysts**")
                st.dataframe(
                    pd.DataFrame(analysts)[["id", "role", "persona", "risk", "gap"]],
                    use_container_width=True,
                    hide_index=True,
                )
        with col_b:
            if aops:
                st.markdown("**AOPs (Areas of Practice)**")
                st.dataframe(
                    pd.DataFrame(aops)[["id", "name", "difficulty"]],
                    use_container_width=True,
                    hide_index=True,
                )


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
        did = d.get("decision_id", "")
        dtype = d.get("decision_type", "")
        payload = d.get("proposed_data") or {}
        ts = d.get("timestamp_proposed", "")
        with st.expander(f"**{dtype}** — {did[:8]}... — {str(ts)[:16]}", expanded=True):
            st.json(payload)
            rationale = st.text_area("Rationale (optional)", key=f"rationale_{did}", placeholder="Reason for approve/reject/modify")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Approve", key=f"approve_{did}"):
                    dm.make_decision(did, DecisionStatus.APPROVED, human_id=expert_name, rationale=rationale)
                    if dtype == "job_task":
                        try:
                            task = JobTask.model_validate({k: v for k, v in payload.items() if k != "_confidence"})
                            sm.add_job_task(task)
                        except Exception:
                            pass
                    elif dtype == "intervention_plan":
                        try:
                            plan = InterventionPlan.model_validate(payload)
                            sm.add_intervention_plan(plan)
                        except Exception:
                            pass
                    st.success("Approved.")
                    st.rerun()
            with col2:
                if st.button("Reject", key=f"reject_{did}"):
                    dm.make_decision(did, DecisionStatus.REJECTED, human_id=expert_name, rationale=rationale)
                    st.info("Rejected.")
                    st.rerun()
            with col3:
                if st.button("Modify", key=f"modify_{did}"):
                    dm.make_decision(did, DecisionStatus.MODIFIED, human_id=expert_name, rationale=rationale, override_data=payload)
                    st.warning("Marked as modified.")
                    st.rerun()


def tab_aop_ingestion() -> None:
    agent = get_ingestion_agent()
    seed = get_seed_data()
    aops = seed.get("aops", [])

    st.subheader("AoP Ingestion (Verb–Task–Product)")
    if aops:
        aop_options = [f"{a['id']} — {a['name']} (difficulty {a.get('difficulty', '?')})" for a in aops]
        aop_choice = st.selectbox("AOP (Area of Practice)", options=["default"] + aop_options, key="aop_ingestion_select")
        aop_id = "default"
        if aop_choice and aop_choice != "default":
            aop_id = aop_choice.split(" — ")[0].strip()
    else:
        aop_id = "default"
    raw = st.text_area("SME design document or raw text", height=200, placeholder="e.g.: Analyze customer feedback → produce weekly report")
    if raw.strip():
        tasks = agent.parse_text(raw.strip(), aop_id=aop_id)
        confidence = [agent.confidence_score(t) for t in tasks]
        for i, (t, c) in enumerate(zip(tasks, confidence)):
            st.markdown(f"**Parsed {i+1}:** `{t.statement}` — confidence: `{c:.2f}`")
        if st.button("Submit to Decision Queue (human validation)"):
            results = agent.ingest_and_submit(raw.strip(), aop_id=aop_id, submit_all=True)
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
    seed = get_seed_data()
    analysts = seed.get("analysts", [])
    aops = seed.get("aops", [])

    st.subheader("Intervention Planner (70:20:10)")

    # Analyst from seed (or manual)
    if analysts:
        analyst_options = [f"{a['id']} — {a.get('role', '')} ({a.get('persona', '')})" for a in analysts]
        analyst_choice = st.selectbox("Analyst (seed)", options=analyst_options, key="planner_analyst_select")
        sel_analyst = analysts[analyst_options.index(analyst_choice)] if analyst_choice else analysts[0]
        analyst_id = sel_analyst["id"]
        # Pre-fill risk/gap from seed (normalize to 0–1: risk as-is if already 0–1, gap: clamp (gap+1)/2 for negative)
        default_risk = max(0.0, min(1.0, float(sel_analyst.get("risk", 0.5))))
        default_gap = max(0.0, min(1.0, (float(sel_analyst.get("gap", 0.5)) + 1) / 2))
    else:
        analyst_id = "default"
        default_risk, default_gap = 0.5, 0.5

    incident_risk = st.slider("Incident risk", 0.0, 1.0, default_risk, 0.1)
    skill_gap = st.slider("Skill gap", 0.0, 1.0, default_gap, 0.1)

    # AOP and difficulty from seed
    if aops:
        aop_options = [f"{a['id']} — {a['name']} (difficulty {a.get('difficulty', '?')})" for a in aops]
        aop_choice = st.selectbox("AOP (Area of Practice)", options=aop_options, key="planner_aop_select")
        sel_aop = aops[aop_options.index(aop_choice)] if aop_choice else aops[0]
        aop_id = sel_aop["id"]
        difficulty = int(sel_aop.get("difficulty", 2))
    else:
        aop_id = "default"
        difficulty = st.selectbox("Difficulty", [1, 2, 3, 4], index=1, format_func=lambda x: f"Level {x}")

    risk_profile = {"incident_risk": incident_risk, "skill_gap": skill_gap}

    if st.button("Generate Plan"):
        plan = planner.generate_plan(
            analyst_id=analyst_id,
            aop_id=aop_id,
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
            for item in _get_interventions(plan.interventions, "OTJ_70"):
                st.write(f"- {item.get('activity', item)}")
        with col2:
            st.markdown("**Social 20%**")
            for item in _get_interventions(plan.interventions, "Social_20"):
                st.write(f"- {item.get('activity', item)}")
        with col3:
            st.markdown("**Formal 10%**")
            for item in _get_interventions(plan.interventions, "Formal_10"):
                st.write(f"- {item.get('activity', item)}")
        if st.button("Submit plan to Decision Queue"):
            dm = get_decision_manager()
            decision_id = dm.propose_decision(
                agent_id=8,
                decision_type="intervention_plan",
                data=plan.model_dump(mode="json"),
                auto_approve=False,
            )
            if "last_plan" in st.session_state:
                del st.session_state["last_plan"]
            st.success(f"Plan submitted for human approval. Decision ID: {decision_id}")
            st.rerun()


def tab_ml_pipeline() -> None:
    """ML Pipeline: synthetic data → feature eng → models → clustering → 70:20:10 plans."""
    st.subheader("ML Pipeline: Feature Eng → Logistic Reg → Clustering → 70:20:10")

    n_analysts = st.number_input("Number of analysts (synthetic)", min_value=20, max_value=120, value=60, step=10, key="n_analysts_pipeline")

    if st.button("Run ML Pipeline", key="run_ml_pipeline"):
        with st.spinner("Running pipeline: synthetic data → feature eng → models → clustering → plans..."):
            try:
                out = run_ml_pipeline(n_analysts=n_analysts)
                st.session_state["ml_pipeline_output"] = out
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return
        st.success("Pipeline complete.")
        st.rerun()

    if "ml_pipeline_output" not in st.session_state:
        st.info("Click **Run ML Pipeline** to generate synthetic data, train models, cluster personas, and build 70:20:10 intervention plans.")
        return

    out = st.session_state["ml_pipeline_output"]
    df = out["df"]
    analyst_features = out["analyst_features"]
    intervention_df = out["intervention_df"]
    intervention_df_full = out["intervention_df_full"]
    model_transfer = out["model_transfer"]
    model_incident = out["model_incident"]
    X = out["X"]

    # Summary metrics
    st.markdown("### Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Analysts", out["n_analysts"])
    col2.metric("AoPs", out["n_aops"])
    col3.metric("Transfer success rate", f"{out['transfer_rate']:.1%}")
    col4.metric("Incident risk rate", f"{out['incident_rate']:.1%}")
    col5.metric("High-risk plans", len(out["intervention_plans"]))

    st.markdown("### Model performance")
    c1, c2 = st.columns(2)
    c1.metric("Transfer success model (AUC)", f"{out['auc_transfer']:.3f}")
    c2.metric("Incident risk model (AUC)", f"{out['auc_incident']:.3f}")

    # Top drivers
    st.markdown("### Top drivers (feature importance)")
    coef_transfer = pd.DataFrame({"Feature": X.columns, "Coef": model_transfer.coef_[0]}).assign(AbsCoef=lambda x: x["Coef"].abs())
    coef_incident = pd.DataFrame({"Feature": X.columns, "Coef": model_incident.coef_[0]}).assign(AbsCoef=lambda x: x["Coef"].abs())
    coef_transfer = coef_transfer.nlargest(8, "AbsCoef")
    coef_incident = coef_incident.nlargest(8, "AbsCoef")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transfer success**")
        fig_t = go.Figure(go.Bar(x=coef_transfer["Coef"], y=coef_transfer["Feature"], orientation="h"))
        fig_t.update_layout(height=300, margin=dict(l=120), xaxis_title="Coefficient")
        st.plotly_chart(fig_t, use_container_width=True)
    with col2:
        st.markdown("**Incident risk**")
        fig_i = go.Figure(go.Bar(x=coef_incident["Coef"], y=coef_incident["Feature"], orientation="h"))
        fig_i.update_layout(height=300, margin=dict(l=120), xaxis_title="Coefficient")
        st.plotly_chart(fig_i, use_container_width=True)

    # Persona distribution
    st.markdown("### Persona distribution")
    persona_counts = out["persona_counts"]
    persona_df = pd.DataFrame({"Persona": persona_counts.index.astype(str), "Count": persona_counts.values})
    st.dataframe(persona_df, use_container_width=True, hide_index=True)

    # Intervention plans (high-risk)
    st.markdown("### Sample intervention plans (high-risk cases)")
    st.dataframe(intervention_df, use_container_width=True, hide_index=True)

    # Charts: risk by persona, 70:20:10 pie, skill gap vs transfer
    st.markdown("### Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig_risk = px.box(df, x="PersonaLabel", y="IncidentRisk_Prob", title="Incident risk by persona")
        fig_risk.update_layout(xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig_risk, use_container_width=True)
    with col2:
        if not intervention_df_full.empty:
            cat_counts = intervention_df_full["Category"].value_counts()
            fig_pie = go.Figure(go.Pie(labels=cat_counts.index, values=cat_counts.values, hole=0.4, marker_colors=["#2E86AB", "#A23B72", "#F18F01"]))
            fig_pie.update_layout(title="Generated intervention mix (70:20:10)", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

    fig_scatter = px.scatter(df.sample(min(500, len(df))), x="SkillGap", y="TransferSuccess_Prob", color="RoleTier", opacity=0.6, title="Skill gap vs predicted transfer success")
    fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5)
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Sample data table
    with st.expander("Sample Analyst × AoP data (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)


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
            interventions = p.interventions or {}
            rows.append({
                "Plan ID": p.plan_id[:8] + "...",
                "Incident risk": rp.get("incident_risk", 0),
                "Skill gap": rp.get("skill_gap", 0),
                "OTJ count": len(_get_interventions(interventions, "OTJ_70")),
                "Social count": len(_get_interventions(interventions, "Social_20")),
                "Formal count": len(_get_interventions(interventions, "Formal_10")),
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dashboard",
        "⚖️ Decision Queue",
        "📥 AoP Ingestion",
        "🎯 Intervention Planner",
        "📈 Analytics",
        "🔬 ML Pipeline",
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
    with tab6:
        tab_ml_pipeline()


if __name__ == "__main__":
    main()
