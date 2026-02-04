"""
Human-in-the-Loop Decision Intelligence for Training Transfer Optimization.
Main Streamlit interface: Dashboard, Decision Queue, AoP Ingestion, Intervention Planner, Analytics.
"""
import sys
import importlib.util
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
from core.state_manager import StateManager
from human_loop.decision_interface import DecisionManager
from agents.agent_01_context_ingestion import ContextIngestionAgent
from agents.agent_08_intervention_planner import InterventionPlannerAgent

# Pipeline import: try package first, then load module from file (for Streamlit Cloud)
try:
    from pipeline.ml_pipeline import run_ml_pipeline, build_intervention_plan_from_row
except ImportError:
    _ml_path = ROOT / "pipeline" / "ml_pipeline.py"
    if _ml_path.exists():
        _spec = importlib.util.spec_from_file_location("ml_pipeline", _ml_path)
        _ml_module = importlib.util.module_from_spec(_spec)
        sys.modules["ml_pipeline"] = _ml_module
        _spec.loader.exec_module(_ml_module)
        run_ml_pipeline = _ml_module.run_ml_pipeline
        build_intervention_plan_from_row = _ml_module.build_intervention_plan_from_row
    else:
        raise


def _get_interventions(interventions: dict, category_value: str) -> list:
    """Get intervention list by category; works with enum or string keys (e.g. after JSON)."""
    if not interventions:
        return []
    try:
        cat = InterventionCategory(category_value)
        return interventions.get(cat, interventions.get(category_value, []))
    except (ValueError, TypeError):
        return interventions.get(category_value, [])


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
    st.sidebar.title("Model Lab")
    st.sidebar.caption("ML pipeline for training transfer & incident risk")
    st.sidebar.divider()
    st.sidebar.markdown("**What you can do:**\n1. Set number of synthetic analysts\n2. Click **Run ML Pipeline**\n3. Explore log, tables, and charts below")
    st.sidebar.divider()
    st.sidebar.caption(f"Expert: **{expert_name}**")


# --- Tabs ---
def tab_dashboard(config: dict) -> None:
    sm = get_state_manager()
    dm = get_decision_manager()
    pending = dm.get_pending_decisions()
    plans = sm.get_intervention_plans()
    tasks = sm.get_job_tasks()
    n_pending = len(pending)

    st.markdown("## Overview Dashboard")
    st.markdown("**Decision Intelligence for Training Transfer** — Target training to what changes performance, align to job tasks and skill gaps, and measure outcomes.")

    with st.expander("📖 What is this app? (Learn as you go)", expanded=False):
        st.markdown("""
This app is a **human-in-the-loop decision system**. It uses data and ML to propose **who** needs **what** training (70:20:10), then **waits for you** to approve or change those proposals. **Nothing goes live until you decide.**

- **Context Library** — See who (analysts) and what (AoPs = job domains) the system knows about.  
- **Data Capture** — Add job tasks in *Verb – Task – Product* form; they are sent to the Decision Queue for your approval.  
- **Decision Queue** — Every AI proposal (job tasks, 70:20:10 plans) appears here. You **Approve**, **Reject**, or **Modify**.  
- **Interventions** — Pick an analyst and AoP, generate a 70:20:10 curriculum, then submit it to the Decision Queue for your approval.  
- **Model Lab** — Run the full ML pipeline on synthetic data to see transfer/incident models, personas, and sample plans.  
- **Reports** — See what you’ve approved (evidence pack).
        """.strip())

    # At a glance: pending callout
    if n_pending > 0:
        st.warning(f"**{n_pending} proposal(s) waiting for your decision.** Go to the **Decision Queue** tab to Approve, Reject, or Modify. Nothing goes live until you decide.")
    else:
        st.success("No pending decisions. Use **Data Capture** to add job tasks, or **Interventions** to generate a 70:20:10 plan—both send proposals to the Decision Queue.")

    st.divider()
    st.markdown("### How this app works (flow)")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
1. **Context Library** — View Analysts and AoPs (Areas of Practice).  
2. **Data Capture** — Paste or type job tasks in *Verb – Task – Product* form; they are parsed and sent to the Decision Queue for your approval.  
3. **Decision Queue** — Review every AI proposal (job tasks, intervention plans). Approve to adopt; Reject or Modify as needed.  
4. **Interventions** — Select an analyst and AoP, set risk/skill gap, then **Generate Plan** to get a 70:20:10 curriculum. Submit the plan to the Queue for approval.  
5. **Model Lab** — Run the full ML pipeline (synthetic data → models → personas → sample plans) to see transfer/incident predictions and persona clusters.  
6. **Reports** — See approved plans and job tasks.
        """.strip())
    with col2:
        labels = ["70:20:10", "OTJ 70%", "Social 20%", "Formal 10%"]
        parents = ["", "70:20:10", "70:20:10", "70:20:10"]
        values = [100, 70, 20, 10]
        fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total"))
        fig.update_layout(title="70:20:10 Learning Model", height=320, margin=dict(t=36, b=16, l=16, r=16))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Pending proposals (high-risk summary)")
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
        st.info("No pending proposals. Create some via **Data Capture** or **Interventions**, then review them in **Decision Queue**.")


def tab_context_library() -> None:
    """Context Library: AoPs, job tasks, skills, KPIs (seed content)."""
    st.markdown("## Context Library")
    st.caption("AoPs (Areas of Practice), Analysts, and seed data. Use this to see who and what the system uses for interventions.")

    with st.expander("📖 What is this phase? How do I use it?", expanded=False):
        st.markdown("""
**What this phase does:**  
This screen shows the **Analysts** and **AoPs (Areas of Practice)** that the system knows about. They come from seed data (`data/seed_data.json`). The **Interventions** tab uses these to build 70:20:10 plans (e.g. “for ANALYST_004, on AoP Information_Gathering”).

**How to use it:**  
- **No action required** to use the app; this is for reference.  
- Check that the analysts and AoPs listed match your use case.  
- When you go to **Interventions**, you will choose one analyst and one AoP from these lists.  
- If the tables are empty, add or fix `data/seed_data.json` with `analysts` and `aops` arrays.
        """.strip())
    st.divider()
    seed = get_seed_data()
    analysts = seed.get("analysts", [])
    aops = seed.get("aops", [])
    if not analysts and not aops:
        st.info("No seed data loaded. Add `data/seed_data.json` with `analysts` and `aops` arrays.")
        return
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Analysts (seed)")
        if analysts:
            st.dataframe(
                pd.DataFrame(analysts)[["id", "role", "persona", "risk", "gap"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("None.")
    with col_b:
        st.markdown("### AoPs — Areas of Practice (seed)")
        if aops:
            st.dataframe(
                pd.DataFrame(aops)[["id", "name", "difficulty"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("None.")


def tab_decision_queue() -> None:
    dm = get_decision_manager()
    sm = get_state_manager()
    config = load_config()
    ui = config.get("ui", {})
    expert_name = ui.get("expert_name", "Dr. Zubia Mughal")

    st.markdown("## Decision Queue")
    st.caption("Proposals from the system wait here. **Approve** to adopt, **Reject** to discard, or **Modify** and record your override. Nothing goes live until you decide.")

    with st.expander("📖 What is this phase? How do I use it?", expanded=False):
        st.markdown("""
**What this phase does:**  
Every AI-generated proposal (new job tasks from **Data Capture**, 70:20:10 plans from **Interventions**) is sent here. **Nothing is adopted until you decide.**

**How to use it:**  
1. Open each proposal (expand the row).  
2. Read the **proposed_data** (JSON).  
3. Optionally add a **Rationale** (e.g. “Approved: matches SME design doc” or “Rejected: duplicate”).  
4. Click **Approve** to adopt the proposal (job task or plan is then available in Reports).  
5. Click **Reject** to discard it.  
6. Click **Modify** to record that you changed it (override is stored with your rationale).  

**Tip:** If you have pending proposals, the sidebar will show a warning and suggest coming here first.
        """.strip())
    st.divider()
    pending = dm.get_pending_decisions()
    if not pending:
        st.success("No pending decisions. Proposals from **Data Capture** (job tasks) and **Interventions** (70:20:10 plans) will appear here for your approval.")
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

    st.markdown("## Data Capture — AoP Context Builder")
    st.caption("Add job tasks in **Verb – Task – Product** form. Parsed tasks are sent to the **Decision Queue** for your approval.")

    with st.expander("📖 What is this phase? How do I use it?", expanded=False):
        st.markdown("""
**What this phase does:**  
You (or SMEs) add **job tasks** that describe what people do in a given AoP (Area of Practice). The app parses them into a standard **Verb – Task – Product** form and sends each task to the **Decision Queue** so you can approve or reject it.

**Format: Verb – Task – Product**  
- **Verb:** action (e.g. Gather, Diagnose, Route).  
- **Task:** what is done (e.g. ticket details for a service call).  
- **Product:** deliverable or outcome (e.g. complete intake in e-automate).  
- Use an arrow **→** or “to produce” between task and product. One task per line works best.

**Sample text to type (copy and paste, then edit):**
```
Gather ticket details for a service call → complete intake in e-automate
Diagnose issue category and severity → routing recommendation using KB
Route service request to correct resolver group → ticket in correct queue
```
**How to use it:**  
1. Choose an **AoP** (optional; links tasks to that area).  
2. Paste or type job tasks (one per line, Verb – Task – Product).  
3. Check the **Parsed tasks (preview)** and confidence scores.  
4. Click **Send to Decision Queue**.  
5. Go to **Decision Queue** to Approve or Reject each task.
        """.strip())
    st.divider()
    if aops:
        aop_options = [f"{a['id']} — {a['name']} (difficulty {a.get('difficulty', '?')})" for a in aops]
        aop_choice = st.selectbox("AoP (Area of Practice) — optional; links tasks to this area", options=["default"] + aop_options, key="aop_ingestion_select")
        aop_id = "default"
        if aop_choice and aop_choice != "default":
            aop_id = aop_choice.split(" — ")[0].strip()
    else:
        aop_id = "default"
    raw = st.text_area(
        "Paste or type job tasks (one per line: Verb – Task – Product). Example: Gather ticket details for a service call → complete intake in e-automate",
        height=220,
        placeholder="Gather ticket details for a service call → complete intake in e-automate\nDiagnose issue category and severity → routing recommendation using KB\nRoute service request to correct resolver group → ticket in correct queue",
    )

    if raw.strip():
        tasks = agent.parse_text(raw.strip(), aop_id=aop_id)
        confidence = [agent.confidence_score(t) for t in tasks]
        st.markdown("**Parsed tasks (preview)** — Review these before sending to the Decision Queue.")
        for i, (t, c) in enumerate(zip(tasks, confidence)):
            st.markdown(f"{i+1}. `{t.statement}` — confidence: {c:.2f}")
    else:
        st.info("Paste or type job tasks above (one per line, **Verb – Task – Product**). Then click **Send to Decision Queue** below and review in **Decision Queue**.")

    if st.button("Send to Decision Queue (requires your approval)", key="data_capture_submit"):
        if not raw.strip():
            st.warning("Enter at least one job task in the text area above, then click **Send to Decision Queue** again.")
        else:
            results = agent.ingest_and_submit(raw.strip(), aop_id=aop_id, submit_all=True)
            for _task, _c, did in results:
                st.success(f"Queued for your approval. Decision ID: {did}. Go to **Decision Queue** to Approve or Reject.")
            st.rerun()


def tab_intervention_planner() -> None:
    planner = get_planner_agent()
    sm = get_state_manager()
    config = load_config()
    ui = config.get("ui", {})
    expert_name = ui.get("expert_name", "Dr. Zubia Mughal")
    seed = get_seed_data()
    analysts = seed.get("analysts", [])
    aops = seed.get("aops", [])

    st.markdown("## Interventions — 70:20:10 Generator")
    st.caption("Select an analyst and AoP, then **Generate Plan** to get a targeted 70:20:10 curriculum (OTJ 70% / Social 20% / Formal 10%). Submit the plan to the **Decision Queue** for your approval.")

    with st.expander("📖 What is this phase? How do I use it?", expanded=False):
        st.markdown("""
**What this phase does:**  
You pick **one analyst** and **one AoP** (from Context Library / seed). The app generates a **70:20:10 plan**:  
- **OTJ 70%** — On-the-job (e.g. structured practice, checklists, performance support).  
- **Social 20%** — Social learning (e.g. peer coaching, critical incident review).  
- **Formal 10%** — Formal training (e.g. eLearning for difficult tasks).  

The plan is **not** applied automatically; you **submit it to the Decision Queue** and then Approve or Reject it there.

**How to use it:**  
1. **Step 1:** Select an analyst (e.g. ANALYST_004 — Analyst, Skill_Builder).  
2. **Step 2:** Select an AoP (e.g. AOP_01 — Information_Gathering, difficulty 2).  
3. **Step 3:** Optionally adjust **Incident risk** (0–1) and **Skill gap** (0–1); they can be pre-filled from seed. Higher values drive more OTJ/Social/Formal activities.  
4. **Step 4:** Click **Generate 70:20:10 Plan**.  
5. **Step 5:** Review the OTJ / Social / Formal breakdown, then click **Submit plan to Decision Queue**.  
6. Go to **Decision Queue** to **Approve** the plan so it appears in **Reports**.
        """.strip())
    st.divider()

    st.markdown("### Step 1: Select analyst")
    if analysts:
        analyst_options = [f"{a['id']} — {a.get('role', '')} ({a.get('persona', '')})" for a in analysts]
        analyst_choice = st.selectbox("Analyst", options=analyst_options, key="planner_analyst_select", label_visibility="collapsed")
        sel_analyst = analysts[analyst_options.index(analyst_choice)] if analyst_choice else analysts[0]
        analyst_id = sel_analyst["id"]
        default_risk = max(0.0, min(1.0, float(sel_analyst.get("risk", 0.5))))
        default_gap = max(0.0, min(1.0, (float(sel_analyst.get("gap", 0.5)) + 1) / 2))
    else:
        analyst_id = "default"
        default_risk, default_gap = 0.5, 0.5
        st.caption("No seed analysts. Using default.")

    st.markdown("### Step 2: Select Area of Practice (AoP)")
    if aops:
        aop_options = [f"{a['id']} — {a['name']} (difficulty {a.get('difficulty', '?')})" for a in aops]
        aop_choice = st.selectbox("AoP", options=aop_options, key="planner_aop_select", label_visibility="collapsed")
        sel_aop = aops[aop_options.index(aop_choice)] if aop_choice else aops[0]
        aop_id = sel_aop["id"]
        difficulty = int(sel_aop.get("difficulty", 2))
    else:
        aop_id = "default"
        difficulty = st.selectbox("Difficulty", [1, 2, 3, 4], index=1, format_func=lambda x: f"Level {x}")

    st.markdown("### Step 3: Set risk profile (optional)")
    st.caption("Higher **Incident risk** (e.g. > 0.6) adds Social 20% activities (e.g. peer coaching). Higher **Skill gap** (e.g. > 0.5) adds OTJ 70% activities. Difficulty ≥ 3 adds Formal 10% (e.g. eLearning).")
    incident_risk = st.slider("Incident risk (0–1)", 0.0, 1.0, default_risk, 0.1)
    skill_gap = st.slider("Skill gap (0–1)", 0.0, 1.0, default_gap, 0.1)
    risk_profile = {"incident_risk": incident_risk, "skill_gap": skill_gap}

    st.markdown("### Step 4: Generate plan")
    if st.button("Generate 70:20:10 Plan"):
        plan = planner.generate_plan(
            analyst_id=analyst_id,
            aop_id=aop_id,
            risk_profile=risk_profile,
            difficulty=difficulty,
        )
        st.session_state["last_plan"] = plan
        st.rerun()

    if "last_plan" in st.session_state:
        plan = st.session_state["last_plan"]
        st.divider()
        st.markdown("### Step 5: Review and submit to Decision Queue")
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
        if st.button("Submit plan to Decision Queue (requires your approval)"):
            dm = get_decision_manager()
            decision_id = dm.propose_decision(
                agent_id=8,
                decision_type="intervention_plan",
                data=plan.model_dump(mode="json"),
                auto_approve=False,
            )
            if "last_plan" in st.session_state:
                del st.session_state["last_plan"]
            st.success(f"Plan sent to **Decision Queue**. Decision ID: {decision_id}. Go to **Decision Queue** to Approve.")
            st.rerun()


def tab_ml_pipeline() -> None:
    """Model Lab: synthetic data → feature eng → models → clustering → 70:20:10 plans."""
    st.markdown("# Model Lab — Train & Validate")
    st.markdown("""
**What you are seeing**  
This is the **ML pipeline** screen. It runs an end-to-end pipeline on **synthetic** data (no real survey data): it generates Analyst × AoP records, builds features, trains two logistic models (Transfer Success and Incident Risk), clusters analysts into personas, and produces sample 70:20:10 intervention plans for high-risk cases.

**How to use it (interactive)**  
- **Number of synthetic analysts** — Choose how many synthetic analyst–AoP records to generate (e.g. 60 or 120).  
- **Run ML Pipeline** — Click to run the pipeline. After it finishes, you’ll see the **Pipeline log**, **Sample Intervention Plans** table, and **Summary visualizations**.  
- You can change the number of analysts and click **Run ML Pipeline** again anytime to regenerate results.
    """)

    with st.expander("📘 Pipeline: Inputs, ML Algorithms, Processing Steps & Dashboard Outputs (detailed)", expanded=True):
        st.markdown("""
---

### 1. INPUTS (what goes into the pipeline)

| Input | Source | Description |
|-------|--------|-------------|
| **Number of synthetic analysts** | **You** (slider on this page) | Integer between 20 and 120. This is the only user-controlled input. It defines how many synthetic “analysts” are created. |
| **Areas of Practice (AoPs)** | **Fixed schema** (inside the pipeline) | 5 AoPs: Information_Gathering, Diagnosis_Triage, Routing_Handoff, Resolution_Documentation, Customer_Communication. Each has an **id**, **name**, **difficulty** (1–4), and **critical_incident_rate** (probability of exposure to critical incidents). |
| **LTSI factors** | **Fixed schema** | 16 Learning Transfer System Inventory (LTSI) factors from the literature, e.g. LearnerReadiness, MotivationToTransfer, PeerSupport, SupervisorSupport, PerformanceSelfEfficacy, TaskCues, etc. They are used to generate synthetic survey-like scores (1–5 scale) per analyst. |
| **Role tiers** | **Random draw** | Each analyst is assigned one of: **Analyst** (50%), **AdvancedAnalyst** (35%), **SeniorAnalyst** (15%). Tier drives base proficiency and LTSI score generation. |

**Synthetic dataset size:** Each analyst is combined with **every** AoP, so total records = **number of analysts × 5**. Example: 60 analysts → 300 rows (Analyst × AoP level).

---

### 2. PROCESSING STEPS & ML ALGORITHMS (what runs behind the screen)

#### Step 1 — Synthetic data generation (no ML)
- For each analyst: assign **RoleTier**, generate **16 LTSI scores** (1–5) with tier-based mean and random noise.
- For each Analyst × AoP pair: compute **SkillGap** (required proficiency from AoP difficulty minus observed proficiency), **CuesAvailable** (beta-distributed), **TransferSuccess** (binary, from a weighted formula using LTSI and gap), **IncidentRisk** (binary, from difficulty, gap, self-efficacy, cues).
- **Output of this step:** One table (dataframe) with one row per Analyst × AoP: AnalystID, RoleTier, AoPID, AoPName, TaskDifficulty, CriticalIncidentFlag, SkillGap, ObservedProficiency, CuesAvailable, TransferSuccess, IncidentRisk, and all 16 LTSI columns.

#### Step 2 — Feature engineering (preprocessing)
- **Base features:** All 16 LTSI factors + TaskDifficulty, SkillGap, CuesAvailable, CriticalIncidentFlag (19 columns).
- **Derived features (created by the pipeline):**
  - **Peer_Supervisor_Interaction** = PeerSupport × SupervisorSupport (interaction term).
  - **Motivation_Efficacy** = MotivationToTransfer × PerformanceSelfEfficacy (interaction term).
  - **High_Difficulty_Low_Cues** = 1 if TaskDifficulty ≥ 3 and CuesAvailable < 0.5, else 0 (binary risk flag).
  - **RoleTier** = numeric encoding (0/1/2) via **LabelEncoder** (sklearn).
- **Scaling:** **StandardScaler** (sklearn) fits on the full feature matrix and transforms it to zero mean, unit variance (required for logistic regression and K-Means).
- **Output of this step:** Feature matrix **X** (scaled), target vectors **y_transfer** (TransferSuccess) and **y_incident** (IncidentRisk).

#### Step 3 — Train/test split
- **Algorithm:** **stratified train_test_split** (sklearn), 80% train / 20% test, **random_state=42**.
- Applied separately for transfer and incident targets so that both models see the same split structure and class balance is preserved in train and test.
- **Output:** X_train, X_test, y_train_transfer, y_test_transfer, y_train_incident, y_test_incident.

#### Step 4 — Binary classification: Transfer Success & Incident Risk (ML algorithm)
- **Algorithm:** **Logistic Regression** (sklearn.linear_model.LogisticRegression).
  - **Max iterations:** 1000.
  - **Class weight:** `"balanced"` (to handle imbalanced 0/1 labels).
  - **Regularization:** L2, strength **C=0.5**.
  - **Solver:** default (lbfgs).
- **Two separate models are trained:**
  1. **Transfer Success model** — predicts probability that training transfers successfully (binary: did transfer happen?). Trained on (X_train, y_train_transfer).
  2. **Incident Risk model** — predicts probability of incident risk (binary: did an incident occur?). Trained on (X_train, y_train_incident).
- **Inference:** Both models run **predict_proba** on the **full** scaled feature matrix to attach to every record: **TransferSuccess_Prob**, **IncidentRisk_Prob**.
- **Evaluation:** **ROC AUC** (sklearn.metrics.roc_auc_score) on the **test** set for each model. These AUC values are shown in the pipeline log and in the final summary block.
- **Output:** Trained `model_transfer` and `model_incident`, plus two probability columns added to the main dataframe.

#### Step 5 — Clustering analysts into personas (ML algorithm)
- **Input:** One row per **analyst** (not per Analyst × AoP): aggregated features (mean of MotivationToTransfer, PerformanceSelfEfficacy, SupervisorSupport, SkillGap, TransferSuccess_Prob, IncidentRisk_Prob, and first RoleTierEncoded).
- **Algorithm:** **K-Means** (sklearn.cluster.KMeans).
  - **Number of clusters:** 3 (fixed).
  - **Random state:** 42.
  - Features are **StandardScaler**-transformed again before clustering.
- **Persona labels (rule-based after clustering):** For each cluster, the pipeline computes average motivation and average skill gap, then assigns a label:
  - **High_Performer** — high motivation (>3.5) and low/negative skill gap.
  - **Needs_Motivation_Support** — low motivation (<3.0).
  - **Skill_Builder** — otherwise (moderate motivation, may have skill gap).
- **Output:** Each analyst (and hence each record) gets **PersonaCluster** (0/1/2) and **PersonaLabel** (High_Performer / Skill_Builder / Needs_Motivation_Support). These are merged back into the main dataframe.

#### Step 6 — Intervention plan generator (rule-based, not ML)
- **Input:** Main dataframe with TransferSuccess_Prob, IncidentRisk_Prob, PersonaLabel, SkillGap, CuesAvailable, CriticalIncidentFlag, SupervisorSupport, TaskDifficulty, AoPName, etc.
- **Selection:** Up to 10 **high-risk** Analyst × AoP pairs where **IncidentRisk_Prob > 0.6** (duplicates removed).
- **Logic:** For each selected pair, a plan is built by **if-then rules**:
  - **OTJ_70:** If SkillGap > 0.5 → add Structured_Practice; if CuesAvailable < 0.5 → add Performance_Support_Tool.
  - **Social_20:** If critical incident or IncidentRisk_Prob > 0.6 → add Peer_Coaching; if SupervisorSupport < 3 → add Manager_Check_in.
  - **Formal_10:** If TaskDifficulty ≥ 3 → add Scenario_Based_eLearning.
- **Output:** List of intervention plans (each with RiskProfile, TargetedInterventions, EvidenceArtifacts, SuccessCriteria). These are flattened into **intervention_df** (summary table) and **intervention_df_full** (one row per intervention activity for the pie chart).

---

### 3. DASHBOARD OUTPUTS (what each part of the screen shows)

| Dashboard element | Data source | Meaning |
|-------------------|-------------|---------|
| **Pipeline log (text block)** | Pipeline run | Step-by-step summary: record count, transfer/incident rates, feature matrix shape, top 5 features correlated with TransferSuccess, **AUC** for both logistic models, **top 5 coefficient** drivers for Transfer Success and Incident Risk, **persona counts**, and a short “export complete” style footer. |
| **Sample Intervention Plans (table)** | intervention_df | One row per high-risk plan: AnalystID, AoPID, Persona, IncidentRisk, SkillGap, counts of OTJ/Social/Formal activities, Artifacts, KPI_Target. This is the **output of Step 6** (rule-based generator). |
| **Incident Risk by Persona Cluster (box plot)** | Main dataframe | **X:** PersonaLabel (High_Performer, Skill_Builder, Needs_Motivation_Support). **Y:** IncidentRisk_Prob (from the **Incident Risk logistic model**). Shows how predicted incident risk varies by persona. |
| **Top Drivers: Training Transfer Success (horizontal bar chart)** | Logistic model coefficients | **Data:** Coefficients of the **Transfer Success** logistic regression (top 8 by absolute value). **Meaning:** Which features push transfer probability up (positive) or down (negative). |
| **Generated Intervention Mix 70:20:10 (pie chart)** | intervention_df_full | **Data:** Count of interventions by category (OTJ_70, Social_20, Formal_10). **Meaning:** Proportion of recommended activities in each bucket (70% OTJ, 20% Social, 10% Formal). |
| **Skill Gap vs Predicted Transfer Success (scatter)** | Main dataframe (sample) | **X:** SkillGap. **Y:** TransferSuccess_Prob (from the **Transfer Success** model). **Color:** RoleTier. Red dashed line at 0.5. Shows how predicted transfer varies with skill gap and role. |
| **Incident Risk — Top drivers (horizontal bar chart)** | Logistic model coefficients | **Data:** Coefficients of the **Incident Risk** logistic regression (top 8). **Meaning:** Which features increase or decrease predicted incident risk. |
| **Sample Analyst × AoP data (expandable table)** | Main dataframe | First 20 rows of the full Analyst × AoP table (all columns). Raw view of **input + model outputs** (probabilities, persona labels) after the pipeline run. |

---

**Summary:** The pipeline uses **two ML algorithms** (Logistic Regression for two binary outcomes, and K-Means for 3 personas), plus **StandardScaler**, **LabelEncoder**, and **stratified train_test_split**. The rest is synthetic data generation and rule-based intervention planning. All dashboard elements are **outputs** of these steps and are fully driven by the pipeline run.
        """)
    st.divider()

    n_analysts = st.number_input("Number of synthetic analysts (e.g. 60 or 120)", min_value=20, max_value=120, value=60, step=10, key="n_analysts_pipeline")

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
    n_records = len(df)
    persona_counts = out["persona_counts"]
    coef_transfer = pd.DataFrame({"Feature": X.columns, "Coef": model_transfer.coef_[0]}).assign(AbsCoef=lambda x: x["Coef"].abs())
    coef_incident = pd.DataFrame({"Feature": X.columns, "Coef": model_incident.coef_[0]}).assign(AbsCoef=lambda x: x["Coef"].abs())
    coef_transfer_top = coef_transfer.nlargest(5, "AbsCoef")
    coef_incident_top = coef_incident.nlargest(5, "AbsCoef")

    # Pipeline log (console-style output)
    st.markdown("### Pipeline log")
    log_lines = [
        "🔧 Generating synthetic dataset matching dissertation schema...",
        "",
        f"✅ Generated {n_records} Analyst × AoP records",
        f"   Transfer Success Rate: {out['transfer_rate']:.2%}",
        f"   Incident Risk Rate: {out['incident_rate']:.2%}",
        "",
        "🔧 Feature Engineering...",
        f"   Feature matrix shape: {X.shape}",
        "   Top features by correlation with TransferSuccess:",
    ]
    corr_with_transfer = X.corrwith(df["TransferSuccess"]).abs().sort_values(ascending=False).head(5)
    for feat, c in corr_with_transfer.items():
        log_lines.append(f"      {feat}: {c:.3f}")
    log_lines.extend([
        "",
        "🤖 Training Logistic Regression Models...",
        "",
        "📊 Model Performance:",
        f"   Transfer Success Model: AUC: {out['auc_transfer']:.3f}",
        f"   Incident Risk Model: AUC: {out['auc_incident']:.3f}",
        "",
        "🔍 Top Drivers (Feature Importance):",
        "",
        "   Transfer Success Top Drivers:",
    ])
    for _, row in coef_transfer_top.iterrows():
        arrow = "↑" if row["Coef"] > 0 else "↓"
        log_lines.append(f"      {arrow} {row['Feature']}: {row['Coef']:.3f}")
    log_lines.extend(["", "   Incident Risk Top Drivers:"])
    for _, row in coef_incident_top.iterrows():
        arrow = "↑" if row["Coef"] > 0 else "↓"
        log_lines.append(f"      {arrow} {row['Feature']}: {row['Coef']:.3f}")
    log_lines.extend([
        "",
        "👥 Clustering Analysts into Personas...",
        "   Persona Distribution:",
    ])
    for persona, count in persona_counts.items():
        log_lines.append(f"   {persona}: {count}")
    log_lines.extend([
        "",
        "🎯 Generating 70:20:10 Intervention Plans...",
        "",
        "📋 Sample Intervention Plans (High Risk Cases):",
    ])
    st.code("\n".join(log_lines), language="text")
    st.dataframe(intervention_df, use_container_width=True, hide_index=True)
    st.code(
        "💾 Exporting structured data for Knowledge Graph...\n"
        "   ✓ kg_analyst_nodes.csv\n"
        "   ✓ kg_aop_nodes.csv\n"
        "   ✓ kg_intervention_edges.csv\n"
        "   ✓ intervention_plans.json\n"
        "\n"
        "📊 Visualization saved: agent_analysis_summary.png\n"
        "\n"
        "=" * 60 + "\n"
        "✅ AGENT PIPELINE COMPLETE\n"
        "=" * 60 + "\n"
        f"• Processed {out['n_analysts']} analysts across {out['n_aops']} AoPs\n"
        f"• Generated {len(out['intervention_plans'])} targeted intervention plans\n"
        f"• Model AUC (Transfer): {out['auc_transfer']:.3f}\n"
        f"• Model AUC (Incident): {out['auc_incident']:.3f}\n"
        f"• Personas identified: {', '.join(persona_counts.index.astype(str))}\n"
        "=" * 60,
        language="text",
    )

    # 2x2 Summary visualization (agent_analysis_summary style)
    st.markdown("### Summary visualization (agent_analysis_summary)")
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        fig_risk = px.box(df, x="PersonaLabel", y="IncidentRisk_Prob", title="Incident Risk by Persona Cluster")
        fig_risk.update_layout(xaxis_tickangle=-45, height=340, margin=dict(t=40, b=40, l=40, r=20))
        st.plotly_chart(fig_risk, use_container_width=True)
    with row1_col2:
        coef_t_8 = coef_transfer.nlargest(8, "AbsCoef")
        fig_t = go.Figure(go.Bar(x=coef_t_8["Coef"], y=coef_t_8["Feature"], orientation="h", marker_color=["#2E86AB" if c > 0 else "#A23B72" for c in coef_t_8["Coef"]]))
        fig_t.update_layout(title="Top Drivers: Training Transfer Success", height=340, margin=dict(t=40, b=40, l=140, r=20), xaxis_title="Coef")
        st.plotly_chart(fig_t, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if not intervention_df_full.empty:
            cat_counts = intervention_df_full["Category"].value_counts()
            fig_pie = go.Figure(go.Pie(labels=cat_counts.index, values=cat_counts.values, hole=0.4, marker_colors=["#2E86AB", "#A23B72", "#F18F01"]))
            fig_pie.update_layout(title="Generated Intervention Mix (70:20:10)", height=340, margin=dict(t=40, b=40, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
    with row2_col2:
        fig_scatter = px.scatter(df.sample(min(500, len(df))), x="SkillGap", y="TransferSuccess_Prob", color="RoleTier", opacity=0.6, title="Skill Gap vs Predicted Transfer Success")
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5)
        fig_scatter.update_layout(height=340, margin=dict(t=40, b=40, l=40, r=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Incident Risk drivers (below 2x2)
    st.markdown("### Incident Risk — Top drivers")
    coef_incident_8 = coef_incident.nlargest(8, "AbsCoef")
    fig_i = go.Figure(go.Bar(x=coef_incident_8["Coef"], y=coef_incident_8["Feature"], orientation="h", marker_color=["#2E86AB" if c > 0 else "#A23B72" for c in coef_incident_8["Coef"]]))
    fig_i.update_layout(height=300, margin=dict(l=140), xaxis_title="Coefficient")
    st.plotly_chart(fig_i, use_container_width=True)

    with st.expander("Sample Analyst × AoP data (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    # --- Targeted intervention plan (based on user selections at each stage) ---
    st.divider()
    st.markdown("## 🎯 Your targeted intervention plan")
    st.markdown("""
The plan below is **driven by your choices at each stage:**
- **Stage 1** — You chose the number of synthetic analysts and ran the pipeline.
- **Stage 2** — Select one **Analyst** and one **Area of Practice (AoP)** from the pipeline results.
- **Stage 3** — The system generates a **single targeted 70:20:10 intervention plan** for that analyst–AoP pair (risk profile, OTJ / Social / Formal activities, evidence artifacts, success criteria).
    """)
    analyst_ids = sorted(df["AnalystID"].unique().tolist())
    aops_list = out.get("aops", [])
    aop_options = [f"{a['id']} — {a['name']} (difficulty {a.get('difficulty', '?')})" for a in aops_list]
    if not aop_options:
        aop_options = [f"{aid}" for aid in sorted(df["AoPID"].unique().tolist())]
        aops_list = [{"id": aid, "name": aid} for aid in sorted(df["AoPID"].unique().tolist())]
    col_a, col_b = st.columns(2)
    with col_a:
        selected_analyst = st.selectbox("Select Analyst (Stage 2)", options=analyst_ids, key="targeted_plan_analyst")
    with col_b:
        selected_aop_label = st.selectbox("Select Area of Practice (AoP) (Stage 2)", options=aop_options, key="targeted_plan_aop")
    selected_aop_id = selected_aop_label.split(" — ")[0].strip() if " — " in selected_aop_label else selected_aop_label
    match = df[(df["AnalystID"] == selected_analyst) & (df["AoPID"] == selected_aop_id)]
    if match.empty:
        st.warning("No pipeline row for this Analyst × AoP pair. Select another combination.")
    else:
        row = match.iloc[0]
        plan = build_intervention_plan_from_row(row, selected_analyst, selected_aop_id)
        rp = plan["RiskProfile"]
        st.markdown("### Stage 3 — Targeted plan output")
        st.markdown(f"**Analyst:** {plan['AnalystID']} · **AoP:** {plan['AoPID']} · **Persona:** {rp['Persona']}")
        st.caption(f"Generated: {plan['GeneratedTimestamp']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Incident risk (prob.)", f"{rp['IncidentRisk_Prob']:.3f}")
        c2.metric("Transfer success (prob.)", f"{rp['TransferSuccess_Prob']:.3f}")
        c3.metric("Skill gap", f"{rp['SkillGap']:.2f}")
        c4.metric("Persona", rp["Persona"])
        st.markdown("#### 70:20:10 interventions")
        otj = plan["TargetedInterventions"]["OTJ_70"]
        social = plan["TargetedInterventions"]["Social_20"]
        formal = plan["TargetedInterventions"]["Formal_10"]
        if otj:
            with st.expander(f"**OTJ (70%)** — {len(otj)} activity(ies)", expanded=True):
                for i, a in enumerate(otj, 1):
                    st.markdown(f"{i}. **{a.get('Type', 'Activity')}** — {a.get('Task', a.get('Topic', '-'))} {a.get('Duration', '')}")
        else:
            st.caption("OTJ (70%): No activities generated for this profile.")
        if social:
            with st.expander(f"**Social (20%)** — {len(social)} activity(ies)", expanded=True):
                for i, a in enumerate(social, 1):
                    st.markdown(f"{i}. **{a.get('Type', 'Activity')}** — {a.get('Topic', a.get('Frequency', '-'))}")
        else:
            st.caption("Social (20%): No activities generated for this profile.")
        if formal:
            with st.expander(f"**Formal (10%)** — {len(formal)} activity(ies)", expanded=True):
                for i, a in enumerate(formal, 1):
                    st.markdown(f"{i}. **{a.get('Type', 'Activity')}** — {a.get('Topic', '-')} {a.get('Duration', '')}")
        else:
            st.caption("Formal (10%): No activities generated for this profile.")
        if plan.get("EvidenceArtifacts"):
            st.markdown("#### Evidence artifacts")
            st.markdown(", ".join(plan["EvidenceArtifacts"]))
        if plan.get("SuccessCriteria"):
            st.markdown("#### Success criteria")
            for k, v in plan["SuccessCriteria"].items():
                st.markdown(f"- **{k}:** {v}")


def tab_analytics() -> None:
    sm = get_state_manager()
    dm = get_decision_manager()
    pending = dm.get_pending_decisions()
    plans = sm.get_intervention_plans()
    tasks = sm.get_job_tasks()

    st.markdown("## Reports — Evidence Pack")
    st.caption("Summary of **approved** intervention plans and job tasks. Use this for evidence packs and exportable summaries.")

    with st.expander("📖 What is this phase? How do I use it?", expanded=False):
        st.markdown("""
**What this phase does:**  
This screen shows what you have **approved** in the **Decision Queue**:  
- **Pending decisions** — Proposals still waiting for your approval (go to Decision Queue to act).  
- **Intervention plans** — 70:20:10 plans you approved (from the Interventions tab).  
- **Job tasks** — Job tasks you approved (from Data Capture).  

**How to use it:**  
- No input required.  
- Use this as your **evidence pack**: what the human expert (you) has signed off on.  
- The **Plans by risk** table lists approved plans with incident risk, skill gap, and OTJ/Social/Formal activity counts.
        """.strip())
    st.divider()
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
        page_title="Model Lab — ML Pipeline",
        page_icon="🔬",
        layout="wide",
    )
    config = load_config()
    render_sidebar(config)
    tab_ml_pipeline()


if __name__ == "__main__":
    main()
