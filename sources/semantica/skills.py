"""
Skill: how the work is carried out.

A skill is a computational graph - sequence, condition and loop - over named
operations. Treating it as data rather than as a Python function is what makes
the fourth kind of learning possible: when feedback shows a step was missing, the
change is a *skill patch* (insert a branch at step 5) rather than a rule change.

Branch conditions use the same condition language as the RBox, evaluated against
a context that falls back to the Situation ABox.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from sources.semantica.conditions import Condition, parse_condition
from sources.semantica.model import AssetKind, Binding, Resolver, Version, utc_now
from sources.semantica.store import SemanticaStore


class SkillContext(Resolver):
    """
    Run-time state of a skill execution.

    Resolution order is context first, then the situation, so a step can shadow
    or enrich a fact (``ctx['vendor_risk'] = 'high'``) and a later branch can
    read it with the same path syntax a rule would use.
    """

    def __init__(self, situation: Optional[Resolver] = None, **values: Any) -> None:
        self.situation = situation
        self.values: Dict[str, Any] = dict(values)
        self.trace: List["StepResult"] = []

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def resolve(self, path: str) -> List[Binding]:
        head = path.split(".")[0]
        if head in self.values:
            value = self.values[head]
            rest = path[len(head) + 1:]
            if not rest:
                return [Binding(path, value, f"ctx:{head}")]
            if isinstance(value, dict):
                cursor: Any = value
                for segment in rest.split("."):
                    if not isinstance(cursor, dict) or segment not in cursor:
                        return []
                    cursor = cursor[segment]
                return [Binding(path, cursor, f"ctx:{head}")]
            return []
        if self.situation is not None:
            return self.situation.resolve(path)
        return []


@dataclass
class StepResult:
    """One entry of the execution trace."""
    step_id: str
    op: str
    status: str = "ok"
    output: Any = None
    note: str = ""
    at: str = field(default_factory=utc_now)

    def describe(self) -> str:
        suffix = f" - {self.note}" if self.note else ""
        return f"[{self.status}] {self.step_id} ({self.op}){suffix}"


class SkillNode:
    """Base class for the skill computational graph."""
    id: str

    def run(self, ctx: SkillContext, runtime: "SkillRuntime") -> Any:
        raise NotImplementedError

    def walk(self) -> Iterator["SkillNode"]:
        yield self

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def outline(self, depth: int = 0) -> List[str]:
        raise NotImplementedError


@dataclass
class Step(SkillNode):
    """A single operation resolved from the runtime registry."""
    id: str
    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    out_key: Optional[str] = None
    optional: bool = False

    def run(self, ctx: SkillContext, runtime: "SkillRuntime") -> Any:
        operation = runtime.registry.get(self.op)
        if operation is None:
            if self.optional:
                ctx.trace.append(StepResult(self.id, self.op, "skipped", note="operation not registered"))
                return None
            raise KeyError(f"skill operation not registered: {self.op}")
        output = operation(ctx, dict(self.params))
        if self.out_key:
            ctx[self.out_key] = output
        ctx.trace.append(StepResult(self.id, self.op, "ok", output))
        return output

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "step", "id": self.id, "op": self.op, "params": self.params,
                "out_key": self.out_key, "optional": self.optional}

    def outline(self, depth: int = 0) -> List[str]:
        return [f"{'  ' * depth}{self.id}: {self.op}"]


@dataclass
class Sequence(SkillNode):
    """Run children in order."""
    id: str
    children: List[SkillNode] = field(default_factory=list)

    def run(self, ctx: SkillContext, runtime: "SkillRuntime") -> Any:
        output = None
        for child in self.children:
            output = child.run(ctx, runtime)
        return output

    def walk(self) -> Iterator[SkillNode]:
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "sequence", "id": self.id, "children": [c.to_dict() for c in self.children]}

    def outline(self, depth: int = 0) -> List[str]:
        lines = [f"{'  ' * depth}{self.id}: sequence"]
        for child in self.children:
            lines.extend(child.outline(depth + 1))
        return lines


@dataclass
class Branch(SkillNode):
    """Condition node: the ``if`` of the computational graph."""
    id: str
    condition: Condition
    then: Optional[SkillNode] = None
    otherwise: Optional[SkillNode] = None

    def run(self, ctx: SkillContext, runtime: "SkillRuntime") -> Any:
        matched = self.condition.evaluate(ctx).matched
        ctx.trace.append(StepResult(self.id, "branch", "ok", matched,
                                    note=f"{self.condition.describe()} -> {matched}"))
        chosen = self.then if matched else self.otherwise
        return chosen.run(ctx, runtime) if chosen is not None else None

    def walk(self) -> Iterator[SkillNode]:
        yield self
        for child in (self.then, self.otherwise):
            if child is not None:
                yield from child.walk()

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "branch", "id": self.id, "condition": self.condition.to_dict(),
                "then": self.then.to_dict() if self.then else None,
                "otherwise": self.otherwise.to_dict() if self.otherwise else None}

    def outline(self, depth: int = 0) -> List[str]:
        lines = [f"{'  ' * depth}{self.id}: if {self.condition.describe()}"]
        if self.then is not None:
            lines.extend(self.then.outline(depth + 1))
        if self.otherwise is not None:
            lines.append(f"{'  ' * depth}else")
            lines.extend(self.otherwise.outline(depth + 1))
        return lines


@dataclass
class Loop(SkillNode):
    """Iterate the body over every value resolved at ``over``."""
    id: str
    over: str
    var: str
    body: Optional[SkillNode] = None
    max_iterations: int = 100

    def run(self, ctx: SkillContext, runtime: "SkillRuntime") -> Any:
        items = [b.value for b in ctx.resolve(self.over)]
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            items = list(items[0])
        outputs = []
        for index, item in enumerate(items[: self.max_iterations]):
            ctx[self.var] = item
            if self.body is not None:
                outputs.append(self.body.run(ctx, runtime))
        ctx.trace.append(StepResult(self.id, "loop", "ok", len(outputs),
                                    note=f"over {self.over} ({len(outputs)} iterations)"))
        return outputs

    def walk(self) -> Iterator[SkillNode]:
        yield self
        if self.body is not None:
            yield from self.body.walk()

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "loop", "id": self.id, "over": self.over, "var": self.var,
                "body": self.body.to_dict() if self.body else None,
                "max_iterations": self.max_iterations}

    def outline(self, depth: int = 0) -> List[str]:
        lines = [f"{'  ' * depth}{self.id}: for {self.var} in {self.over}"]
        if self.body is not None:
            lines.extend(self.body.outline(depth + 1))
        return lines


def parse_node(data: Dict[str, Any]) -> SkillNode:
    """Rebuild a skill node from its serialised form."""
    kind = data.get("kind")
    if kind == "step":
        return Step(id=data["id"], op=data["op"], params=dict(data.get("params", {})),
                    out_key=data.get("out_key"), optional=bool(data.get("optional", False)))
    if kind == "sequence":
        return Sequence(id=data["id"], children=[parse_node(c) for c in data.get("children", [])])
    if kind == "branch":
        return Branch(id=data["id"], condition=parse_condition(data["condition"]),
                      then=parse_node(data["then"]) if data.get("then") else None,
                      otherwise=parse_node(data["otherwise"]) if data.get("otherwise") else None)
    if kind == "loop":
        return Loop(id=data["id"], over=data["over"], var=data["var"],
                    body=parse_node(data["body"]) if data.get("body") else None,
                    max_iterations=int(data.get("max_iterations", 100)))
    raise ValueError(f"unknown skill node kind: {kind!r}")


@dataclass
class SkillRun:
    """Result of executing a skill."""
    skill_id: str
    version: str
    ctx: SkillContext
    trace: List[StepResult]

    def outline(self) -> str:
        return "\n".join(step.describe() for step in self.trace)


class SkillGraph:
    """A versioned skill: an executable computational graph."""

    def __init__(self, id: str, name: str, root: SkillNode, version: str = "1.0",
                 description: str = "") -> None:
        self.id = id
        self.name = name
        self.root = root
        self.version = str(Version.parse(version))
        self.description = description

    def steps(self) -> List[SkillNode]:
        return list(self.root.walk())

    def find(self, node_id: str) -> Optional[SkillNode]:
        return next((n for n in self.root.walk() if n.id == node_id), None)

    def outline(self) -> str:
        return "\n".join(self.root.outline())

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "version": self.version,
                "description": self.description, "root": self.root.to_dict()}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SkillGraph":
        return SkillGraph(id=data["id"], name=data.get("name", data["id"]),
                          root=parse_node(data["root"]), version=data.get("version", "1.0"),
                          description=data.get("description", ""))

    def copy(self) -> "SkillGraph":
        return SkillGraph.from_dict(copy.deepcopy(self.to_dict()))

    def insert_after(self, step_id: str, node: SkillNode) -> "SkillGraph":
        """Derive the next skill version with ``node`` inserted after ``step_id``."""
        evolved = self.copy()
        evolved.version = str(Version.parse(self.version).bump_minor())
        if not _insert_after(evolved.root, step_id, node):
            raise KeyError(f"step not found in skill graph: {step_id}")
        return evolved

    def replace(self, step_id: str, node: SkillNode) -> "SkillGraph":
        """Derive the next skill version with ``step_id`` replaced by ``node``."""
        evolved = self.copy()
        evolved.version = str(Version.parse(self.version).bump_minor())
        if not _replace(evolved.root, step_id, node):
            raise KeyError(f"step not found in skill graph: {step_id}")
        return evolved

    def save(self, store: SemanticaStore, changelog: str = "",
             parent_version: Optional[str] = None) -> None:
        store.put_asset(AssetKind.SKILL, self.id, self.version, self.to_dict(),
                        changelog=changelog, parent_version=parent_version)

    @staticmethod
    def load(store: SemanticaStore, skill_id: str, version: Optional[str] = None) -> Optional["SkillGraph"]:
        record = store.get_asset(AssetKind.SKILL, skill_id, version)
        return SkillGraph.from_dict(record.payload) if record is not None else None


def _insert_after(node: SkillNode, step_id: str, new_node: SkillNode) -> bool:
    if isinstance(node, Sequence):
        for index, child in enumerate(node.children):
            if child.id == step_id:
                node.children.insert(index + 1, new_node)
                return True
        return any(_insert_after(child, step_id, new_node) for child in node.children)
    if isinstance(node, Branch):
        return any(_insert_after(child, step_id, new_node)
                   for child in (node.then, node.otherwise) if child is not None)
    if isinstance(node, Loop) and node.body is not None:
        return _insert_after(node.body, step_id, new_node)
    return False


def _replace(node: SkillNode, step_id: str, new_node: SkillNode) -> bool:
    if isinstance(node, Sequence):
        for index, child in enumerate(node.children):
            if child.id == step_id:
                node.children[index] = new_node
                return True
        return any(_replace(child, step_id, new_node) for child in node.children)
    if isinstance(node, Branch):
        if node.then is not None and node.then.id == step_id:
            node.then = new_node
            return True
        if node.otherwise is not None and node.otherwise.id == step_id:
            node.otherwise = new_node
            return True
        return any(_replace(child, step_id, new_node)
                   for child in (node.then, node.otherwise) if child is not None)
    if isinstance(node, Loop) and node.body is not None:
        if node.body.id == step_id:
            node.body = new_node
            return True
        return _replace(node.body, step_id, new_node)
    return False


class SkillRuntime:
    """Executes skill graphs against a registry of named operations."""

    def __init__(self, registry: Optional[Dict[str, Callable[[SkillContext, Dict[str, Any]], Any]]] = None) -> None:
        self.registry: Dict[str, Callable[[SkillContext, Dict[str, Any]], Any]] = dict(registry or {})

    def register(self, op: str, fn: Callable[[SkillContext, Dict[str, Any]], Any]) -> None:
        self.registry[op] = fn

    def run(self, skill: SkillGraph, ctx: SkillContext) -> SkillRun:
        skill.root.run(ctx, self)
        return SkillRun(skill_id=skill.id, version=skill.version, ctx=ctx, trace=list(ctx.trace))


@dataclass
class SkillChangeProposal:
    """
    A proposed edit to the computational graph.

    Emitted when feedback shows the *procedure* was wrong rather than the rules -
    for example, an external vendor request needing a risk assessment before the
    security check runs.
    """
    id: str
    skill_id: str
    base_version: str
    operation: str  # "insert_after" | "replace"
    anchor_step_id: str
    node: SkillNode
    rationale: str = ""
    evidence: Tuple[str, ...] = ()
    status: str = "proposed"
    created_at: str = field(default_factory=utc_now)

    def apply(self, skill: SkillGraph) -> SkillGraph:
        if self.operation == "insert_after":
            evolved = skill.insert_after(self.anchor_step_id, self.node)
        elif self.operation == "replace":
            evolved = skill.replace(self.anchor_step_id, self.node)
        else:
            raise ValueError(f"unknown skill patch operation: {self.operation}")
        self.status = "applied"
        return evolved

    def describe(self) -> str:
        return (f"Skill proposal {self.id} [{self.status}]: {self.operation} "
                f"{self.node.id} at {self.anchor_step_id} of {self.skill_id} "
                f"v{self.base_version}\n  rationale: {self.rationale}")

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "skill_id": self.skill_id, "base_version": self.base_version,
                "operation": self.operation, "anchor_step_id": self.anchor_step_id,
                "node": self.node.to_dict(), "rationale": self.rationale,
                "evidence": list(self.evidence), "status": self.status,
                "created_at": self.created_at}
