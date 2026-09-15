"""
RBox: what must happen when.

A rule is a condition over the Situation ABox plus the action it demands. Rules
are stored as data and versioned, decisions are resolved by "most restrictive
wins", and every decision carries an evidence graph that names the facts and the
rules behind it - an auditor can replay it without the agent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sources.semantica.conditions import Condition, ConditionResult, parse_condition
from sources.semantica.model import (AssetKind, Binding, Edge, Graph, Node, Provenance,
                                     Resolver, Version, utc_now)
from sources.semantica.store import SemanticaStore


class Action(str, Enum):
    """Possible outcomes, ordered from least to most restrictive."""
    AUTO_APPROVE = "AUTO_APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    SECURITY_REVIEW = "SECURITY_REVIEW"
    REJECT = "REJECT"

    @property
    def severity(self) -> int:
        order = [Action.AUTO_APPROVE, Action.MANUAL_REVIEW, Action.SECURITY_REVIEW, Action.REJECT]
        return order.index(self)

    @staticmethod
    def coerce(value: Any) -> "Action":
        if isinstance(value, Action):
            return value
        return Action(str(value).strip().upper())


@dataclass
class Rule:
    """
    One policy statement.

    Attributes:
        id: Stable identifier, e.g. ``R102``.
        version: Version of this rule; amendments bump it rather than overwrite.
        when: Condition over the Situation ABox.
        action: What the rule demands when it fires.
        scope: Optional gate, e.g. only for ``FirewallChangeRequest``.
        status: ``active``, ``proposed`` or ``deprecated``.
        source: Where the rule came from - a policy document or a learned delta.
    """
    id: str
    when: Condition
    action: Action
    version: str = "1.0"
    priority: int = 100
    scope: Optional[Condition] = None
    rationale: str = ""
    status: str = "active"
    tags: Tuple[str, ...] = ()
    source: str = ""

    def applies_to(self, resolver: Resolver) -> bool:
        return True if self.scope is None else self.scope.evaluate(resolver).matched

    def evaluate(self, resolver: Resolver) -> ConditionResult:
        return self.when.evaluate(resolver)

    def describe(self) -> str:
        return (f"{self.id}-v{self.version}: IF {self.when.describe()} "
                f"THEN {self.action.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "version": self.version, "when": self.when.to_dict(),
            "action": self.action.value, "priority": self.priority,
            "scope": self.scope.to_dict() if self.scope else None,
            "rationale": self.rationale, "status": self.status,
            "tags": list(self.tags), "source": self.source,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Rule":
        return Rule(
            id=data["id"], when=parse_condition(data["when"]),
            action=Action.coerce(data["action"]), version=data.get("version", "1.0"),
            priority=int(data.get("priority", 100)),
            scope=parse_condition(data["scope"]) if data.get("scope") else None,
            rationale=data.get("rationale", ""), status=data.get("status", "active"),
            tags=tuple(data.get("tags", ())), source=data.get("source", ""),
        )

    def copy(self) -> "Rule":
        return Rule.from_dict(copy.deepcopy(self.to_dict()))


@dataclass
class RuleHit:
    """A rule that fired, with the bindings that made it fire."""
    rule_id: str
    version: str
    action: Action
    evidence: Tuple[Binding, ...]
    rationale: str = ""

    def describe(self) -> str:
        facts = "; ".join(f"{b.path}={b.value!r}" for b in self.evidence) or "no bindings"
        return f"{self.rule_id}-v{self.version} -> {self.action.value} ({facts})"


@dataclass
class Decision:
    """The recommendation plus everything needed to defend it."""
    case_id: str
    action: Action
    hits: Tuple[RuleHit, ...]
    rbox_version: str
    missing_paths: Tuple[str, ...] = ()
    evidence_graph: Graph = field(default_factory=Graph)
    decided_at: str = field(default_factory=utc_now)

    @property
    def driving_hit(self) -> Optional[RuleHit]:
        """The hit that determined the outcome."""
        candidates = [h for h in self.hits if h.action == self.action]
        return candidates[0] if candidates else None

    def explain(self) -> str:
        lines = [f"decision: {self.action.value} (RBox v{self.rbox_version})"]
        for hit in self.hits:
            lines.append(f"  - {hit.describe()}")
        if not self.hits:
            lines.append("  - no rule fired, default action applied")
        if self.missing_paths:
            lines.append(f"  ! unresolved facts: {', '.join(sorted(set(self.missing_paths)))}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id, "action": self.action.value,
            "rbox_version": self.rbox_version, "decided_at": self.decided_at,
            "missing_paths": list(self.missing_paths),
            "hits": [{"rule_id": h.rule_id, "version": h.version, "action": h.action.value,
                      "evidence": [{"path": b.path, "value": b.value, "node_id": b.node_id}
                                   for b in h.evidence]} for h in self.hits],
            "evidence_graph": self.evidence_graph.to_dict(),
        }


class RBox:
    """A versioned rule set."""

    def __init__(self, name: str, version: str = "1.0", rules: Optional[Iterable[Rule]] = None,
                 default_action: Action = Action.AUTO_APPROVE) -> None:
        self.name = name
        self.version = str(Version.parse(version))
        self.rules: List[Rule] = list(rules or [])
        self.default_action = default_action

    def get(self, rule_id: str) -> Optional[Rule]:
        return next((r for r in self.rules if r.id == rule_id), None)

    def active_rules(self) -> List[Rule]:
        return sorted([r for r in self.rules if r.status == "active"], key=lambda r: r.priority)

    def applicable(self, resolver: Resolver) -> List[Rule]:
        return [r for r in self.active_rules() if r.applies_to(resolver)]

    def evaluate(self, situation: Any) -> Decision:
        """
        Evaluate every applicable rule and resolve to one action.

        Conflicts are resolved by severity: the most restrictive firing rule
        wins, so a new exception can only ever tighten an outcome, never loosen
        one silently.
        """
        hits: List[RuleHit] = []
        missing: List[str] = []
        for rule in self.applicable(situation):
            result = rule.evaluate(situation)
            missing.extend(result.missing)
            if result.matched:
                hits.append(RuleHit(rule.id, rule.version, rule.action, result.evidence,
                                    rule.rationale))
        action = max((h.action for h in hits), key=lambda a: a.severity) if hits else self.default_action
        hits.sort(key=lambda h: h.action.severity, reverse=True)
        case_id = getattr(situation, "case_id", "case")
        decision = Decision(case_id=case_id, action=action, hits=tuple(hits),
                            rbox_version=self.version, missing_paths=tuple(missing))
        decision.evidence_graph = build_evidence_graph(decision, situation)
        return decision

    def with_rule(self, rule: Rule, changelog: str = "") -> "RBox":
        """Derive the next RBox version carrying an added or amended rule."""
        evolved = RBox(self.name, str(Version.parse(self.version).bump_minor()),
                       [r.copy() for r in self.rules], self.default_action)
        existing = evolved.get(rule.id)
        if existing is not None:
            evolved.rules = [r for r in evolved.rules if r.id != rule.id]
            superseded = existing.copy()
            superseded.status = "deprecated"
            superseded.id = f"{existing.id}@{existing.version}"
            evolved.rules.append(superseded)
        evolved.rules.append(rule.copy())
        return evolved

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version,
                "default_action": self.default_action.value,
                "rules": [r.to_dict() for r in self.rules]}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RBox":
        return RBox(name=data["name"], version=data.get("version", "1.0"),
                    rules=[Rule.from_dict(r) for r in data.get("rules", [])],
                    default_action=Action.coerce(data.get("default_action", "AUTO_APPROVE")))

    def save(self, store: SemanticaStore, changelog: str = "",
             parent_version: Optional[str] = None) -> None:
        store.put_asset(AssetKind.RBOX, self.name, self.version, self.to_dict(),
                        changelog=changelog, parent_version=parent_version)

    @staticmethod
    def load(store: SemanticaStore, name: str, version: Optional[str] = None) -> Optional["RBox"]:
        record = store.get_asset(AssetKind.RBOX, name, version)
        return RBox.from_dict(record.payload) if record is not None else None


def build_evidence_graph(decision: Decision, situation: Any) -> Graph:
    """
    Materialise the decision as a graph: Decision -> Rule -> Fact -> Entity.

    Keeping the evidence as a graph rather than a string is what lets the same
    artefact be shown to a reviewer, replayed by a back-test and stored next to
    the case in Semantica.
    """
    graph = Graph(name=f"evidence:{decision.case_id}", root_id=f"decision:{decision.case_id}")
    decision_id = f"decision:{decision.case_id}"
    graph.add_node(Node(id=decision_id, type="Decision", ontology="evidence",
                        props={"action": decision.action.value,
                               "rbox_version": decision.rbox_version,
                               "decided_at": decision.decided_at},
                        provenance=Provenance(source="rbox", asset_version=decision.rbox_version)))
    situation_graph = getattr(situation, "graph", None)
    for hit in decision.hits:
        rule_node_id = f"rule:{hit.rule_id}@{hit.version}"
        graph.add_node(Node(id=rule_node_id, type="Rule", ontology="evidence",
                            props={"action": hit.action.value, "rationale": hit.rationale},
                            provenance=Provenance(source="rbox", asset_version=hit.version)))
        graph.add_edge(Edge(decision_id, "supportedBy", rule_node_id, ontology="evidence"))
        for binding in hit.evidence:
            fact_id = f"fact:{hit.rule_id}:{binding.path}={binding.value}"
            graph.add_node(Node(id=fact_id, type="Fact", ontology="evidence",
                                props={"path": binding.path, "value": binding.value},
                                provenance=Provenance(source=binding.ontology or "situation")))
            graph.add_edge(Edge(rule_node_id, "matchedOn", fact_id, ontology="evidence"))
            if situation_graph is not None and binding.node_id in situation_graph.nodes:
                entity = situation_graph.nodes[binding.node_id]
                graph.add_node(entity)
                graph.add_edge(Edge(fact_id, "assertedOn", entity.id, ontology="evidence"))
    return graph
