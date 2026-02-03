# Human-in-the-Loop Decision Intelligence for Training Transfer Optimization

Production-grade Python application for **training transfer optimization** based on the **70:20:10 learning model**, with a governance layer that queues AI proposals for human approval (never auto-approves interventions).

---

## What This App Is Doing (System Overview)

**In one sentence:** This is a **human-in-the-loop decision system** that uses data and ML to propose **who** needs **what** training (70:20:10), then **waits for you** to approve or change those proposals before anything “goes live.”

### The Big Idea

- **Problem:** Training is often generic; time is wasted on things that don’t move performance, and real job tasks, skill gaps, and risk aren’t targeted.
- **Approach:** Ingest **context** (job tasks, analysts, AoPs). Use **ML** to predict transfer success and incident risk, cluster people into personas, and suggest 70:20:10 plans. Put every AI suggestion into a **Decision Queue** so a human expert (you) approves, rejects, or modifies it. Nothing is adopted until you decide.

**Flow:** Context → Data capture → Models (optional) → Interventions → **Decision Queue (your approval)** → Reports.

### What Each Part Does

| Part | Purpose |
|------|--------|
| **Overview** | Start here. See the 70:20:10 model, “how this app works,” and what’s waiting on you (pending proposals). |
| **Context Library** | See **who** (analysts) and **what** (AoPs = Areas of Practice) the system knows about. Seed data defines the “universe” the app reasons about. |
| **Data Capture** | You (or SMEs) paste or type **job tasks** in **Verb – Task – Product** form. The app parses them and **sends them to the Decision Queue** for your approval. |
| **Decision Queue** | **Every** AI-generated proposal (new job tasks, 70:20:10 plans) lands here. You **Approve** (adopt), **Reject** (discard), or **Modify** (override + rationale). Approved items feed into Reports. |
| **Interventions** | Pick one analyst and one AoP, set risk/skill gap, then **Generate Plan** to get a 70:20:10 curriculum. Submit the plan to the **Decision Queue** for your approval. |
| **Model Lab** | Run the full ML pipeline on **synthetic** data: transfer/incident models, personas, sample 70:20:10 plans. Proves the pipeline; same flow will work with real data later. |
| **Reports** | See what’s been **approved**: counts and lists of intervention plans and job tasks (evidence pack). |

### Governance and Targeting

- **Governance:** The app never commits a job task or an intervention plan by itself. It **proposes**; you **decide**. That’s the “human-in-the-loop” and “decision intelligence” in practice.
- **Targeting:** Suggestions are tied to **analyst + AoP + risk/gap/difficulty**, so plans are “who needs what” rather than generic.
- **Traceability:** The Decision Queue plus rationale and audit-style logging give a record of **who** approved what and **why**, for evidence packs and rigor.

---

## Project Structure

```
dr_data_agents/
├── agents/           # 9 specialized agents (01 context ingestion, 08 intervention planner implemented)
├── human_loop/       # Decision interface + audit trail
├── core/             # Pydantic schema, state manager
├── pipeline/         # ML pipeline (synthetic data → models → clustering → plans)
├── ui/               # Streamlit app
├── config/           # config.yaml (governance, thresholds)
├── data/             # input, seed_data.json, pending_decisions, kg
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Core Requirements

- **Schema (core/schema.py):** Pydantic models: `DifficultyLevel`, `SkillType`, `ProficiencyLevel`, `InterventionCategory`, `JobTask`, `HumanDecision`, `InterventionPlan`.
- **Human-in-the-Loop (human_loop/decision_interface.py):** `DecisionManager` — `propose_decision()`, `get_pending_decisions()`, `make_decision()`; auto-save to JSON; never auto-approve.
- **Agent 01:** Parse SME design documents into `JobTask` (Verb–Task–Product), confidence scoring, submit to human validation.
- **Agent 08:** Generate 70:20:10 plans from `risk_profile` (incident_risk, skill_gap) and difficulty.
- **Streamlit UI:** Overview, Context Library, Data Capture, Decision Queue, Interventions, Model Lab, Reports.

## Run Locally

1. **From this directory (repo root):**
   ```bash
   pip install -r requirements.txt
   streamlit run ui/streamlit_app.py
   ```

2. **Optional (install as package from parent of this folder):**
   ```bash
   cd path/to/parent
   pip install -e dr_data_agents
   streamlit run dr_data_agents/ui/streamlit_app.py
   ```

## Publish on Streamlit Cloud

1. Push this project to a **new GitHub repository** (recommended: avoid OneDrive-synced paths).
2. In [Streamlit Community Cloud](https://share.streamlit.io), connect the repo.
3. Set **Main file path** to `ui/streamlit_app.py`.
4. Set **Root directory** to the repo root.
5. Deploy. Streamlit Cloud will install from `requirements.txt` and run the app.

## Config

Edit `config/config.yaml` for:

- `governance_mode`, `decision_authority`, `auto_approve_interventions` (must stay `false`)
- Agent thresholds (e.g. `skill_gap_threshold`, `incident_risk_threshold`)
- Paths and UI expert name/role

## Tech Stack

- **streamlit** — UI  
- **pydantic** — data validation  
- **plotly** — visualizations  
- **pandas** — data manipulation  
- **pyyaml** — config  
- **scikit-learn** — models and clustering  

## License

Use under your chosen license.
