# Human-in-the-Loop Decision Intelligence for Training Transfer Optimization

Production-grade Python application for **training transfer optimization** based on the **70:20:10 learning model**, with a governance layer that queues AI proposals for human approval (never auto-approves interventions).

## Project structure

```
dr_data_agents/
├── agents/           # 9 specialized agents (01 context ingestion, 08 intervention planner implemented)
├── human_loop/      # Decision interface + audit trail
├── core/            # Pydantic schema, state manager
├── ui/              # Streamlit app
├── config/          # config.yaml (governance, thresholds)
├── data/            # input, pending_decisions, kg
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Core requirements

- **Schema (core/schema.py):** Pydantic models: `DifficultyLevel`, `SkillType`, `ProficiencyLevel`, `InterventionCategory`, `JobTask`, `HumanDecision`, `InterventionPlan` (70:20:10 curriculum map with evidence artifacts).
- **Human-in-the-Loop (human_loop/decision_interface.py):** `DecisionManager` — `propose_decision()`, `get_pending_decisions()`, `make_decision()`; auto-save to JSON; never auto-approve.
- **Agent 01:** Parse SME design documents into `JobTask` (Verb–Task–Product), confidence scoring, submit to human validation.
- **Agent 08:** Generate 70:20:10 plans from `risk_profile` (incident_risk, skill_gap) and difficulty; return `InterventionPlan` with evidence artifacts.
- **Streamlit UI:** Dashboard (70:20:10 sunburst, high-risk table), Decision Queue (Approve/Reject/Modify), AoP Ingestion, Intervention Planner, Analytics.

## Run locally

1. **From this directory (repo root):**
   ```bash
   pip install -r requirements.txt
   streamlit run ui/streamlit_app.py
   ```
   If the repo root is the folder that contains `agents/`, `core/`, `human_loop/`, `ui/`, then the app adds this directory to `sys.path` so imports work.

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
4. Set **Root directory** to the repo root (the folder containing `ui/`, `core/`, etc.).
5. Deploy. Streamlit Cloud will install from `requirements.txt` and run the app.

## Config

Edit `config/config.yaml` for:

- `governance_mode`, `decision_authority`, `auto_approve_interventions` (must stay `false`)
- Agent thresholds (e.g. `skill_gap_threshold`, `incident_risk_threshold`)
- Paths and UI expert name/role

## Tech stack

- **streamlit** — UI  
- **pydantic** — data validation  
- **plotly** — visualizations (e.g. sunburst)  
- **pandas** — data manipulation  
- **pyyaml** — config  
- **pathlib** — file operations  

## License

Use under your chosen license.
