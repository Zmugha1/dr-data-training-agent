# Human-in-the-Loop Decision Intelligence for Training Transfer Optimization

Production-grade Python application for **training transfer optimization** based on the **70:20:10 learning model**, with a governance layer that queues AI proposals for human approval (never auto-approves interventions).

---

## Project Description & Goals (The Story)

### Context: Why This Exists

In many organizations, **training is designed and delivered without a clear line of sight to real job tasks**. People sit through courses, but back on the job the “transfer climate” is weak: support is misaligned, difficult tasks lack performance support, and roughly **one in three** training efforts fail to transfer into better performance. That gap wastes L&D hours and leaves risk on the table—especially in high-stakes roles like service-desk analysts, where complex technical workflows meet demanding customer interactions. Critical incidents happen when people lack the right cues, confidence, or support at the moment of need—not only when they lack “knowledge” from a class.

This project starts from that reality: **training transfer is a system problem**, not just a content problem. It asks: *Can we use data and models to say who needs what kind of support (on-the-job, social, or formal), and then put a human expert in charge of every decision so nothing goes live without approval?*

### The Problem in Plain Language

- **Generic training** doesn’t target who is actually at risk or where the real skill gaps are.
- **Transfer fails** when motivation and confidence (e.g. “I can do this”) don’t line up with support (e.g. peers and managers aligned).
- **Incident risk** is often tied to difficult tasks, missing cues, and low self-efficacy—not just “did they attend training?”
- **One-size plans** waste resources on people who don’t need them and under-support those who do.

So we need a system that (1) **connects** job tasks, skills, and risk in one place, (2) **predicts** who is likely to transfer (or not) and who is at higher incident risk, (3) **segments** people into actionable personas, and (4) **proposes** targeted 70:20:10 plans—while (5) **never** auto-applying anything until a human expert approves it.

### What the System Does (Story Form)

1. **Data and context**  
   The system works with **analysts** and **Areas of Practice (AoPs)**—the real tasks and workflows (e.g. information gathering, diagnosis & triage, routing). It uses factors from learning-transfer research (e.g. motivation to transfer, self-efficacy, peer and supervisor support) and turns them into features. **Skill gap** is the gap between what a task requires and what someone can do today—a core driver of both transfer failure and incident risk.

2. **Models that explain, not just predict**  
   Two models sit at the heart: one for **transfer success** (will learning stick?) and one for **incident risk** (how likely is a critical incident?). They are kept interpretable (e.g. logistic regression) so we can see *why*—e.g. “motivation × efficacy” and “skill gap” drive transfer; difficulty and cues drive risk. The system uses these predictions to **prioritize** who gets attention and to **govern** how much to rely on each signal (e.g. weight transfer more for resource allocation, use incident risk as a watch flag).

3. **Personas that map to how people learn**  
   People are grouped into three broad personas:
   - **Needs Motivation Support** — capable but low confidence or motivation; need visible support (e.g. peer coaching, manager check-ins) and structured practice.
   - **Skill Builder** — motivated but hitting complexity ceilings on hard tasks; need performance support (cues, job aids), scenario-based learning, and critical-incident review.
   - **High Performer** — low gap, high efficacy; need minimal intervention and can act as mentors in the “Social 20%” for others.

   That split drives **who gets which mix** of 70% on-the-job, 20% social, and 10% formal—so the right people get the right kind of help.

4. **Interventions with evidence and KPIs**  
   For each at-risk analyst–task pair, the system suggests a **targeted 70:20:10 plan**: OTJ activities (e.g. checklists, performance support tools), social activities (e.g. critical-incident review, peer coaching), and formal activities (e.g. scenario-based eLearning for difficult tasks). Each plan is tied to **evidence artifacts** (e.g. observation checklists, incident analysis tables) and **success criteria** (e.g. “50% incident reduction in 30 days” for highest risk). So every recommendation is traceable and measurable.

5. **Human in the loop**  
   No job task and no intervention plan is adopted by the system alone. Every proposal goes into a **Decision Queue** where a human expert (e.g. L&D or operations) **approves, rejects, or modifies** it and can add a rationale. That creates an audit trail and keeps governance first—exactly what’s needed for evidence-based practice and stakeholder trust.

### Goals (What We’re Aiming For)

| Goal | Meaning |
|------|--------|
| **Reduce wasted training** | Target only what changes performance: the right person, the right task, the right mix of 70:20:10. |
| **Improve transfer** | Align learning and support to real job tasks, critical incidents, and skill gaps so transfer climate and confidence both improve. |
| **Improve operational KPIs** | Tie interventions to measurable outcomes (e.g. incident rate, proficiency deltas) and document them for evidence packs. |
| **Standardize SME collaboration** | Turn expert knowledge (job tasks, difficulty, cues, errors) into a repeatable, data-driven workflow that still leaves final decisions to humans. |
| **Operationalize the research** | Turn discovery questions, design documents, LTSI-style factors, and proficiency rubrics into one pipeline: data → models → personas → 70:20:10 plans → human approval → reports. |

### How the Pipeline Fits the Story

The **Model Lab** in this app runs a full pipeline on synthetic data that mirrors the above: it generates analyst–task records, builds features (including motivation–efficacy and skill gap), trains the two models, clusters personas, and produces sample 70:20:10 plans with evidence artifacts and KPI targets. The same flow is designed to run later on **real** survey and checklist data. So the app is both a **proof of concept** and a **governance-first** demo: you see what the system would recommend, and you stay in control of what actually gets applied.

**Summary in one line:** This project turns “who needs what training and why” into a clear, auditable, human-approved process—so training transfer and incident risk are addressed in a targeted, evidence-based way instead of by one-size-fits-all programs.

---

*If you have a dissertation or longer report that describes the theoretical framework, instruments, or case context in more detail, sharing it can help refine this narrative and align the app’s language and logic even more closely with your research.*

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
