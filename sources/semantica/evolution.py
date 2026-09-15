"""
Learning as verifiable evolution of cognitive assets.

When the human decision differs from the recommendation, that mismatch is the
training signal. It is not used to fine-tune anything; it is used to *propose* a
change to an asset, which is then back-tested against history and approved by a
human before a new version exists:

    case feedback -> decision mismatch -> root cause -> gap detection
        -> patch proposal -> back-test against historical cases
        -> human approval -> RBox v2

The LLM never edits a production rule set. It can only produce proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sources.semantica.abox import SituationABox
from sources.semantica.conditions import All, Any_, Cmp, Condition
from sources.semantica.model import Edge, Node, Provenance, utc_now
from sources.semantica.rbox import Action, Decision, RBox, Rule
from sources.semantica.store import SemanticaStore


@dataclass
class CaseRecord:
    """A closed case: what the system said, what the human did."""
    case_id: str
    situation: SituationABox
    system_action: Action
    human_action: Action
    human_reason: str = ""
    rbox_version: str = ""
    closed_at: str = field(default_factory=utc_now)

    @property
    def is_mismatch(self) -> bool:
        return self.system_action != self.human_action


class CaseArchive:
    """History of closed cases; the regression suite for rule learning."""

    def __init__(self, cases: Optional[Iterable[CaseRecord]] = None) -> None:
        self.cases: List[CaseRecord] = list(cases or [])

    def add(self, case: CaseRecord) -> CaseRecord:
        self.cases.append(case)
        return case

    def mismatches(self) -> List[CaseRecord]:
        return [c for c in self.cases if c.is_mismatch]

    def replay(self, rbox: RBox) -> Dict[str, Action]:
        """Evaluate every archived situation against a rule set."""
        return {case.case_id: rbox.evaluate(case.situation).action for case in self.cases}

    def __len__(self) -> int:
        return len(self.cases)


@dataclass
class DecisionDelta:
    """The gap between recommendation and human action."""
    case_id: str
    system_action: Action
    human_action: Action
    reason: str
    situation: SituationABox
    decision: Optional[Decision] = None
    observed_at: str = field(default_factory=utc_now)

    @property
    def is_tightening(self) -> bool:
        """True when the human was more restrictive than the system."""
        return self.human_action.severity > self.system_action.severity

    def describe(self) -> str:
        return (f"Decision delta on {self.case_id}\n"
                f"  system recommendation: {self.system_action.value}\n"
                f"  human decision: {self.human_action.value}\n"
                f"  reason: {self.reason}")


@dataclass
class BackTestReport:
    """What a proposal would have done to history."""
    total: int = 0
    fixed: Tuple[str, ...] = ()
    regressed: Tuple[str, ...] = ()
    unchanged: Tuple[str, ...] = ()

    @property
    def is_safe(self) -> bool:
        return not self.regressed

    def describe(self) -> str:
        return (f"back-test over {self.total} cases: "
                f"fixed={len(self.fixed)} regressed={len(self.regressed)} "
                f"unchanged={len(self.unchanged)}"
                + (f"\n  regressions: {', '.join(self.regressed)}" if self.regressed else ""))

    def to_dict(self) -> Dict[str, Any]:
        return {"total": self.total, "fixed": list(self.fixed),
                "regressed": list(self.regressed), "unchanged": list(self.unchanged)}


@dataclass
class RuleChangeProposal:
    """
    A proposed rule amendment or addition.

    Nothing here is live: ``proposed_rule`` only becomes policy through
    ``GovernanceGate.approve``, which bumps the RBox version.
    """
    id: str
    kind: str  # "amend" | "new"
    proposed_rule: Rule
    base_rule_id: Optional[str] = None
    base_rule: Optional[Rule] = None
    rationale: str = ""
    evidence: Tuple[str, ...] = ()
    backtest: Optional[BackTestReport] = None
    status: str = "proposed"
    created_at: str = field(default_factory=utc_now)
    approved_by: str = ""

    def describe(self) -> str:
        lines = [f"Rule proposal {self.id} [{self.status}] ({self.kind})"]
        if self.base_rule is not None:
            lines.append(f"  from: {self.base_rule.describe()}")
        lines.append(f"  to:   {self.proposed_rule.describe()}")
        if self.rationale:
            lines.append(f"  rationale: {self.rationale}")
        if self.evidence:
            lines.append(f"  evidence: {', '.join(self.evidence)}")
        if self.backtest is not None:
            lines.append(f"  {self.backtest.describe()}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "base_rule_id": self.base_rule_id,
                "proposed_rule": self.proposed_rule.to_dict(),
                "base_rule": self.base_rule.to_dict() if self.base_rule else None,
                "rationale": self.rationale, "evidence": list(self.evidence),
                "backtest": self.backtest.to_dict() if self.backtest else None,
                "status": self.status, "created_at": self.created_at,
                "approved_by": self.approved_by}


def candidate_paths(situation: SituationABox, max_hops: int = 2) -> List[str]:
    """
    Enumerate the paths a rule could read from a situation.

    This is the hypothesis space for counterfactual analysis: everything the
    situation knows, including facts no current rule looks at.
    """
    graph = situation.graph
    root = situation.root_id
    paths: List[str] = []

    def add(path: str) -> None:
        if path not in paths:
            paths.append(path)

    root_node = graph.get(root)
    if root_node is not None:
        for key in sorted(root_node.props):
            if key != "label":
                add(key)
    frontier: List[Tuple[str, str]] = [(root, "")]
    for _ in range(max_hops):
        next_frontier: List[Tuple[str, str]] = []
        for node_id, prefix in frontier:
            for edge in graph.out_edges(node_id):
                path = f"{prefix}{edge.rel}" if not prefix else f"{prefix}{edge.rel}"
                add(path)
                target = graph.get(edge.dst)
                if target is None:
                    continue
                add(f"{path}.#type")
                for key in sorted(target.props):
                    if key != "label":
                        add(f"{path}.{key}")
                next_frontier.append((edge.dst, f"{path}."))
        frontier = next_frontier
    return paths


class CounterfactualAnalyzer:
    """
    Root-cause a decision mismatch and propose the minimal rule change.

    The question asked for every candidate fact is: *if the rule had also looked
    at this, would it have produced the human's decision without changing any
    other case in history?*
    """

    def __init__(self, archive: CaseArchive, max_hops: int = 2,
                 ignore_paths: Sequence[str] = ()) -> None:
        self.archive = archive
        self.max_hops = max_hops
        self.ignore_paths = set(ignore_paths)

    def analyze(self, delta: DecisionDelta, rbox: RBox,
                max_proposals: int = 3) -> List[RuleChangeProposal]:
        hypotheses = self._hypotheses(delta)
        target_rule = self._target_rule(delta, rbox)
        scored: List[Tuple[Tuple[int, int, int], RuleChangeProposal]] = []
        tester = BackTester(self.archive)
        for path, hypothesis in hypotheses:
            if target_rule is not None:
                proposed = amend_rule(target_rule, hypothesis, delta.situation)
                kind, base = "amend", target_rule
            else:
                proposed = new_rule(delta, hypothesis)
                kind, base = "new", None
            candidate_rbox = rbox.with_rule(proposed)
            report = tester.run(candidate_rbox, extra=[_delta_as_case(delta)])
            if not report.fixed or report.regressed:
                continue
            proposal = RuleChangeProposal(
                id=f"rule-proposal-{delta.case_id}-{path.replace('.', '-')}",
                kind=kind,
                proposed_rule=proposed,
                base_rule_id=base.id if base else None,
                base_rule=base,
                rationale=(f"{delta.reason or 'human decision differed'} "
                           f"-> explained by {hypothesis.describe()}"),
                evidence=tuple(dict.fromkeys([delta.case_id, *report.fixed])),
                backtest=report,
            )
            scored.append(((len(report.regressed), -len(report.fixed), len(path)), proposal))
        scored.sort(key=lambda item: item[0])
        return [proposal for _, proposal in scored[:max_proposals]]

    def _hypotheses(self, delta: DecisionDelta) -> List[Tuple[str, Cmp]]:
        hypotheses: List[Tuple[str, Cmp]] = []
        for path in candidate_paths(delta.situation, self.max_hops):
            if path in self.ignore_paths:
                continue
            for binding in delta.situation.resolve(path):
                if isinstance(binding.value, (dict, list)):
                    continue
                hypotheses.append((path, Cmp(path, "eq", binding.value)))
        return hypotheses

    def _target_rule(self, delta: DecisionDelta, rbox: RBox) -> Optional[Rule]:
        """
        The rule that should have fired.

        Prefer an active rule demanding exactly the human's action whose
        condition is partially satisfied - that is the one with a gap in it.
        """
        best: Optional[Rule] = None
        best_score = -1.0
        for rule in rbox.active_rules():
            if rule.action != delta.human_action:
                continue
            score = _partial_match_score(rule.when, delta.situation)
            if score > best_score:
                best, best_score = rule, score
        return best if best is not None and best_score > 0 else None


def _partial_match_score(condition: Condition, situation: SituationABox) -> float:
    """Fraction of a conjunction's sub-conditions that already hold."""
    if isinstance(condition, All) and condition.conditions:
        matched = sum(1 for c in condition.conditions if c.evaluate(situation).matched)
        return matched / len(condition.conditions)
    return 1.0 if condition.evaluate(situation).matched else 0.0


def amend_rule(rule: Rule, hypothesis: Condition, situation: SituationABox) -> Rule:
    """
    Widen the sub-condition that blocked the rule.

    For ``All([A, B])`` where B failed, the amendment is ``All([A, Any([B, H])])``
    - the shape a security engineer would write by hand when they say "and also
    when the requester is an external vendor".
    """
    amended = rule.copy()
    amended.version = _bump(rule.version)
    amended.status = "active"
    amended.source = "learned"
    when = amended.when
    if isinstance(when, All) and when.conditions:
        blocking = [index for index, child in enumerate(when.conditions)
                    if not child.evaluate(situation).matched]
        if blocking:
            index = blocking[0]
            children = list(when.conditions)
            children[index] = Any_([children[index], hypothesis])
            amended.when = All(children)
            return amended
    amended.when = Any_([when, hypothesis])
    return amended


def new_rule(delta: DecisionDelta, hypothesis: Condition) -> Rule:
    """Propose a brand new rule when no existing rule covers the human action."""
    root = delta.situation.root
    scope = Cmp("#type", "eq", root.type) if root is not None else None
    return Rule(
        id=f"R-learned-{delta.case_id}",
        when=All([hypothesis]),
        action=delta.human_action,
        version="1.0",
        priority=50,
        scope=scope,
        rationale=delta.reason or "learned from a human decision",
        status="active",
        tags=("learned",),
        source=f"delta:{delta.case_id}",
    )


class BackTester:
    """Replays a candidate rule set over closed cases."""

    def __init__(self, archive: CaseArchive) -> None:
        self.archive = archive

    def run(self, candidate: RBox, extra: Sequence[CaseRecord] = ()) -> BackTestReport:
        """
        Args:
            candidate: The rule set that would exist if the proposal were approved.
            extra: Cases not yet in the archive, typically the case that produced
                the delta being analysed.
        """
        fixed: List[str] = []
        regressed: List[str] = []
        unchanged: List[str] = []
        for case in [*self.archive.cases, *extra]:
            before_ok = case.system_action == case.human_action
            after = candidate.evaluate(case.situation).action
            after_ok = after == case.human_action
            if not before_ok and after_ok:
                fixed.append(case.case_id)
            elif before_ok and not after_ok:
                regressed.append(case.case_id)
            else:
                unchanged.append(case.case_id)
        total = len(self.archive.cases) + len(extra)
        return BackTestReport(total=total, fixed=tuple(fixed), regressed=tuple(regressed),
                              unchanged=tuple(unchanged))


class GovernanceGate:
    """
    The control that keeps the system learnable but not self-modifying.

    A proposal must carry a back-test and an approver before it can produce a new
    RBox version; an unsafe proposal can only pass with an explicit override that
    is recorded in the changelog.
    """

    def __init__(self, archive: CaseArchive, store: Optional[SemanticaStore] = None) -> None:
        self.archive = archive
        self.store = store
        self.pending: List[RuleChangeProposal] = []
        self.history: List[RuleChangeProposal] = []

    def submit(self, proposal: RuleChangeProposal, rbox: RBox) -> RuleChangeProposal:
        if proposal.backtest is None:
            proposal.backtest = BackTester(self.archive).run(rbox.with_rule(proposal.proposed_rule))
        proposal.status = "submitted"
        self.pending.append(proposal)
        if self.store is not None:
            record_proposal(self.store, "rbox", proposal.id, proposal.to_dict())
        return proposal

    def approve(self, proposal: RuleChangeProposal, rbox: RBox, approver: str,
                override_reason: str = "") -> RBox:
        if not approver:
            raise PermissionError("a rule change requires a named approver")
        if proposal.backtest is None:
            raise PermissionError(f"{proposal.id} has not been back-tested")
        if not proposal.backtest.is_safe and not override_reason:
            raise PermissionError(
                f"{proposal.id} regresses {len(proposal.backtest.regressed)} historical cases; "
                "an explicit override reason is required")
        evolved = rbox.with_rule(proposal.proposed_rule)
        proposal.status = "approved"
        proposal.approved_by = approver
        self.pending = [p for p in self.pending if p.id != proposal.id]
        self.history.append(proposal)
        changelog = (f"{proposal.id} approved by {approver}: {proposal.rationale}"
                     + (f" [override: {override_reason}]" if override_reason else ""))
        if self.store is not None:
            evolved.save(self.store, changelog=changelog, parent_version=rbox.version)
            record_proposal(self.store, "rbox", proposal.id, proposal.to_dict())
        return evolved

    def reject(self, proposal: RuleChangeProposal, approver: str, reason: str) -> RuleChangeProposal:
        proposal.status = "rejected"
        proposal.approved_by = approver
        proposal.rationale = f"{proposal.rationale} | rejected: {reason}"
        self.pending = [p for p in self.pending if p.id != proposal.id]
        self.history.append(proposal)
        if self.store is not None:
            record_proposal(self.store, "rbox", proposal.id, proposal.to_dict())
        return proposal


def record_proposal(store: SemanticaStore, asset_kind: str, proposal_id: str,
                    payload: Dict[str, Any]) -> Node:
    """
    Keep every proposed delta in Semantica, whatever asset it targets.

    ΔTBox, ΔABox, ΔRBox and ΔSkill all land in the same governance ontology, so
    the evolution of the agent's mind is itself a queryable graph.
    """
    node = Node(id=f"proposal:{proposal_id}", type="ChangeProposal", ontology="governance",
                props={"asset_kind": asset_kind, "status": payload.get("status", "proposed"),
                       "rationale": payload.get("rationale", ""),
                       "created_at": payload.get("created_at", utc_now())},
                provenance=Provenance(source="evolution", derived_from=tuple(payload.get("evidence", ()))))
    store.upsert_node(node)
    for evidence_id in payload.get("evidence", ()):
        store.upsert_edge(Edge(node.id, "evidencedBy", f"case:{evidence_id}", ontology="governance"))
    return node


def _delta_as_case(delta: DecisionDelta) -> CaseRecord:
    return CaseRecord(case_id=delta.case_id, situation=delta.situation,
                      system_action=delta.system_action, human_action=delta.human_action,
                      human_reason=delta.reason)


def _bump(version: str) -> str:
    major, _, minor = version.partition(".")
    return f"{major}.{int(minor) + 1}"
