"""
Condition language shared by the RBox and by skill branching.

Conditions are *data*, not code: they serialise to JSON, so a rule can be
proposed, back-tested, versioned and approved without anyone editing Python. Each
evaluation returns the bindings that made it true, which is what the evidence
graph is built from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sources.semantica.model import Binding, Resolver


@dataclass
class ConditionResult:
    """Outcome of evaluating a condition, with the facts that decided it."""
    matched: bool
    evidence: Tuple[Binding, ...] = ()
    missing: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.matched


class Condition:
    """Base class for the condition AST."""

    def evaluate(self, resolver: Resolver) -> ConditionResult:
        raise NotImplementedError

    def paths(self) -> List[str]:
        """Every path the condition reads; used by gap detection and learning."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


_NUMERIC_OPS = {"gt", "gte", "lt", "lte"}


@dataclass
class Cmp(Condition):
    """Compare the values at a path with an expected value."""
    path: str
    op: str = "eq"
    value: Any = None

    def evaluate(self, resolver: Resolver) -> ConditionResult:
        bindings = resolver.resolve(self.path)
        if self.op == "exists":
            return ConditionResult(bool(bindings), tuple(bindings))
        if self.op == "missing":
            return ConditionResult(not bindings, tuple(bindings))
        if not bindings:
            return ConditionResult(False, (), (self.path,))
        hits = [b for b in bindings if self._compare(b.value)]
        if self.op in ("ne", "not_in"):
            # Negative operators hold only when the path resolves and nothing matches.
            return ConditionResult(len(hits) == len(bindings), tuple(bindings))
        return ConditionResult(bool(hits), tuple(hits))

    def _compare(self, actual: Any) -> bool:
        if self.op == "eq":
            return actual == self.value
        if self.op == "ne":
            return actual != self.value
        if self.op == "in":
            return actual in (self.value or [])
        if self.op == "not_in":
            return actual not in (self.value or [])
        if self.op == "contains":
            try:
                return self.value in actual
            except TypeError:
                return False
        if self.op in _NUMERIC_OPS:
            left, right = _coerce(actual), _coerce(self.value)
            if left is None or right is None:
                return False
            if self.op == "gt":
                return left > right
            if self.op == "gte":
                return left >= right
            if self.op == "lt":
                return left < right
            return left <= right
        raise ValueError(f"unknown operator: {self.op}")

    def paths(self) -> List[str]:
        return [self.path]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "cmp", "path": self.path, "op": self.op, "value": self.value}

    def describe(self) -> str:
        if self.op in ("exists", "missing"):
            return f"{self.path} {self.op}"
        symbol = {"eq": "=", "ne": "!=", "in": "in", "not_in": "not in", "contains": "contains",
                  "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}.get(self.op, self.op)
        return f"{self.path} {symbol} {self.value!r}"


@dataclass
class All(Condition):
    """Every sub-condition must hold."""
    conditions: Sequence[Condition] = field(default_factory=list)

    def evaluate(self, resolver: Resolver) -> ConditionResult:
        evidence: List[Binding] = []
        missing: List[str] = []
        for condition in self.conditions:
            result = condition.evaluate(resolver)
            missing.extend(result.missing)
            if not result.matched:
                return ConditionResult(False, tuple(evidence), tuple(missing))
            evidence.extend(result.evidence)
        return ConditionResult(True, tuple(evidence), tuple(missing))

    def paths(self) -> List[str]:
        return [p for c in self.conditions for p in c.paths()]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "all", "conditions": [c.to_dict() for c in self.conditions]}

    def describe(self) -> str:
        return " AND ".join(f"({c.describe()})" for c in self.conditions)


@dataclass
class Any_(Condition):
    """At least one sub-condition must hold."""
    conditions: Sequence[Condition] = field(default_factory=list)

    def evaluate(self, resolver: Resolver) -> ConditionResult:
        missing: List[str] = []
        for condition in self.conditions:
            result = condition.evaluate(resolver)
            missing.extend(result.missing)
            if result.matched:
                return ConditionResult(True, result.evidence, tuple(missing))
        return ConditionResult(False, (), tuple(missing))

    def paths(self) -> List[str]:
        return [p for c in self.conditions for p in c.paths()]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "any", "conditions": [c.to_dict() for c in self.conditions]}

    def describe(self) -> str:
        return " OR ".join(f"({c.describe()})" for c in self.conditions)


@dataclass
class Not(Condition):
    """Negation, keeping the inspected bindings as evidence."""
    condition: Optional[Condition] = None

    def evaluate(self, resolver: Resolver) -> ConditionResult:
        result = self.condition.evaluate(resolver)
        return ConditionResult(not result.matched, result.evidence, result.missing)

    def paths(self) -> List[str]:
        return self.condition.paths()

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "not", "condition": self.condition.to_dict()}

    def describe(self) -> str:
        return f"NOT ({self.condition.describe()})"


@dataclass
class Always(Condition):
    """Constant condition, useful as a rule default or an unconditional branch."""
    value: bool = True

    def evaluate(self, resolver: Resolver) -> ConditionResult:
        return ConditionResult(self.value)

    def paths(self) -> List[str]:
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "always", "value": self.value}

    def describe(self) -> str:
        return "always" if self.value else "never"


def parse_condition(data: Dict[str, Any]) -> Condition:
    """Rebuild a condition from its serialised form."""
    kind = data.get("kind")
    if kind == "cmp":
        return Cmp(path=data["path"], op=data.get("op", "eq"), value=data.get("value"))
    if kind == "all":
        return All([parse_condition(c) for c in data.get("conditions", [])])
    if kind == "any":
        return Any_([parse_condition(c) for c in data.get("conditions", [])])
    if kind == "not":
        return Not(parse_condition(data["condition"]))
    if kind == "always":
        return Always(bool(data.get("value", True)))
    raise ValueError(f"unknown condition kind: {kind!r}")


def _coerce(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
