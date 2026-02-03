import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from core.schema import HumanDecision, DecisionStatus

class DecisionManager:
    def __init__(self, pending_dir: str = "./data/pending_decisions"):
        self.pending_dir = Path(pending_dir)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        
    def propose_decision(self, agent_id: int, decision_type: str, 
                        data: Dict[str, Any], auto_approve: bool = False) -> str:
        decision_id = f"DEC_{agent_id:02d}_{uuid.uuid4().hex[:8]}"
        
        decision = HumanDecision(
            decision_id=decision_id,
            agent_id=agent_id,
            decision_type=decision_type,
            proposed_data=data,
            status=DecisionStatus.APPROVED if auto_approve else DecisionStatus.PENDING,
            timestamp_proposed=datetime.now()
        )
        
        if not auto_approve:
            self._save_decision(decision)
            print(f"⏸️  AWAITING HUMAN DECISION: {decision_id}")
        return decision_id
    
    def get_pending_decisions(self) -> List[Dict]:
        pending = []
        for file in self.pending_dir.glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                if data['status'] == 'pending':
                    pending.append(data)
        return sorted(pending, key=lambda x: x['timestamp_proposed'])
    
    def make_decision(self, decision_id: str, status: DecisionStatus, 
                     human_id: str, override_data: Optional[Dict] = None,
                     rationale: Optional[str] = None):
        decision_file = self.pending_dir / f"{decision_id}.json"
        
        if not decision_file.exists():
            raise ValueError(f"Decision {decision_id} not found")
            
        with open(decision_file) as f:
            decision = json.load(f)
            
        decision['status'] = status.value
        decision['human_expert_id'] = human_id
        decision['timestamp_decided'] = datetime.now().isoformat()
        decision['rationale'] = rationale
        
        if override_data:
            decision['human_override'] = override_data
            
        decided_file = self.pending_dir / "decided" / f"{decision_id}.json"
        decided_file.parent.mkdir(exist_ok=True)
        
        with open(decided_file, 'w') as f:
            json.dump(decision, f, indent=2, default=str)
            
        decision_file.unlink()
        return decision
    
    def _save_decision(self, decision: HumanDecision):
        file_path = self.pending_dir / f"{decision.decision_id}.json"
        with open(file_path, 'w') as f:
            json.dump(decision.model_dump(), f, indent=2, default=str)
