"""
TBox: the world model.

The TBox says which concepts may exist and what may be said about them. It is not
designed once at boot: when a case cannot be represented with the current
vocabulary, the gap detector emits a *proposal* rather than silently editing the
schema, and only a validated proposal produces a new TBox version.

    case -> ontology gap -> TBox proposal -> validation -> new TBox version
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sources.semantica.model import AssetKind, Graph, Node, Version, utc_now
from sources.semantica.store import SemanticaStore


@dataclass(frozen=True)
class PropertySpec:
    """A literal attribute a concept may carry."""
    name: str
    datatype: str = "string"
    required: bool = False
    allowed: Tuple[Any, ...] = ()
    description: str = ""

    def validate(self, value: Any) -> Optional[str]:
        if self.allowed and value not in self.allowed:
            return f"value {value!r} not in allowed set {list(self.allowed)}"
        checks = {"int": int, "float": (int, float), "bool": bool, "string": str}
        expected = checks.get(self.datatype)
        if expected is not None and not isinstance(value, expected):
            return f"value {value!r} is not of datatype {self.datatype}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "datatype": self.datatype, "required": self.required,
                "allowed": list(self.allowed), "description": self.description}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PropertySpec":
        return PropertySpec(name=data["name"], datatype=data.get("datatype", "string"),
                            required=bool(data.get("required", False)),
                            allowed=tuple(data.get("allowed", ())),
                            description=data.get("description", ""))


@dataclass(frozen=True)
class RelationSpec:
    """A typed edge a concept may take part in."""
    name: str
    target: str
    required: bool = False
    functional: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "target": self.target, "required": self.required,
                "functional": self.functional, "description": self.description}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RelationSpec":
        return RelationSpec(name=data["name"], target=data.get("target", "Thing"),
                            required=bool(data.get("required", False)),
                            functional=bool(data.get("functional", True)),
                            description=data.get("description", ""))


@dataclass
class ConceptDef:
    """A class in the world model."""
    name: str
    parents: Tuple[str, ...] = ()
    properties: Dict[str, PropertySpec] = field(default_factory=dict)
    relations: Dict[str, RelationSpec] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parents": list(self.parents),
            "properties": {k: v.to_dict() for k, v in self.properties.items()},
            "relations": {k: v.to_dict() for k, v in self.relations.items()},
            "description": self.description,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ConceptDef":
        return ConceptDef(
            name=data["name"],
            parents=tuple(data.get("parents", ())),
            properties={k: PropertySpec.from_dict(v) for k, v in data.get("properties", {}).items()},
            relations={k: RelationSpec.from_dict(v) for k, v in data.get("relations", {}).items()},
            description=data.get("description", ""),
        )


class TBox:
    """A versioned set of concept definitions."""

    def __init__(self, name: str, version: str = "1.0", concepts: Optional[Iterable[ConceptDef]] = None) -> None:
        self.name = name
        self.version = str(Version.parse(version))
        self.concepts: Dict[str, ConceptDef] = {c.name: c for c in (concepts or [])}

    # --- lookup ----------------------------------------------------------
    def has_concept(self, name: str) -> bool:
        return name in self.concepts

    def ancestors(self, name: str) -> List[str]:
        seen: List[str] = []
        queue = list(self.concepts.get(name, ConceptDef(name)).parents)
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.append(current)
            queue.extend(self.concepts.get(current, ConceptDef(current)).parents)
        return seen

    def is_a(self, name: str, ancestor: str) -> bool:
        return name == ancestor or ancestor in self.ancestors(name)

    def properties_of(self, name: str) -> Dict[str, PropertySpec]:
        merged: Dict[str, PropertySpec] = {}
        for concept_name in reversed(self.ancestors(name)):
            merged.update(self.concepts.get(concept_name, ConceptDef(concept_name)).properties)
        merged.update(self.concepts.get(name, ConceptDef(name)).properties)
        return merged

    def relations_of(self, name: str) -> Dict[str, RelationSpec]:
        merged: Dict[str, RelationSpec] = {}
        for concept_name in reversed(self.ancestors(name)):
            merged.update(self.concepts.get(concept_name, ConceptDef(concept_name)).relations)
        merged.update(self.concepts.get(name, ConceptDef(name)).relations)
        return merged

    # --- validation ------------------------------------------------------
    def validate_node(self, node: Node, graph: Optional[Graph] = None) -> List[str]:
        """Return the list of violations for one node (empty list means valid)."""
        if not self.has_concept(node.type):
            return [f"{node.id}: unknown concept '{node.type}'"]
        violations: List[str] = []
        specs = self.properties_of(node.type)
        for key, value in node.props.items():
            if key == "label":
                continue
            spec = specs.get(key)
            if spec is None:
                violations.append(f"{node.id}: property '{key}' is not declared on '{node.type}'")
                continue
            problem = spec.validate(value)
            if problem is not None:
                violations.append(f"{node.id}.{key}: {problem}")
        for key, spec in specs.items():
            if spec.required and key not in node.props:
                violations.append(f"{node.id}: required property '{key}' is missing")
        if graph is not None:
            declared = self.relations_of(node.type)
            for edge in graph.out_edges(node.id):
                if edge.rel not in declared:
                    violations.append(f"{node.id}: relation '{edge.rel}' is not declared on '{node.type}'")
            for key, relation in declared.items():
                if relation.required and not graph.out_edges(node.id, key):
                    violations.append(f"{node.id}: required relation '{key}' is missing")
        return violations

    def validate_graph(self, graph: Graph) -> List[str]:
        violations: List[str] = []
        for node in graph.walk():
            violations.extend(self.validate_node(node, graph))
        return violations

    # --- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version,
                "concepts": {k: v.to_dict() for k, v in self.concepts.items()}}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TBox":
        return TBox(name=data["name"], version=data.get("version", "1.0"),
                    concepts=[ConceptDef.from_dict(c) for c in data.get("concepts", {}).values()])

    def save(self, store: SemanticaStore, changelog: str = "", parent_version: Optional[str] = None) -> None:
        store.put_asset(AssetKind.TBOX, self.name, self.version, self.to_dict(),
                        changelog=changelog, parent_version=parent_version)

    @staticmethod
    def load(store: SemanticaStore, name: str, version: Optional[str] = None) -> Optional["TBox"]:
        record = store.get_asset(AssetKind.TBOX, name, version)
        return TBox.from_dict(record.payload) if record is not None else None

    def copy(self) -> "TBox":
        return TBox.from_dict(copy.deepcopy(self.to_dict()))


@dataclass
class TBoxEvolutionProposal:
    """
    A proposed extension of the world model.

    This is the artefact a case produces when the ontology cannot express what
    the case needs - for example "temporary vendor access" having no concept.
    Nothing changes until it is validated and applied.
    """
    id: str
    observed_gap: str
    proposed_concept: Optional[ConceptDef] = None
    added_properties: Dict[str, List[PropertySpec]] = field(default_factory=dict)
    added_relations: Dict[str, List[RelationSpec]] = field(default_factory=dict)
    evidence: Tuple[str, ...] = ()
    rationale: str = ""
    status: str = "proposed"
    created_at: str = field(default_factory=utc_now)

    def apply(self, tbox: TBox, changelog: Optional[str] = None) -> TBox:
        """Derive the next TBox version; the input TBox is left untouched."""
        evolved = tbox.copy()
        evolved.version = str(Version.parse(tbox.version).bump_minor())
        if self.proposed_concept is not None:
            evolved.concepts[self.proposed_concept.name] = copy.deepcopy(self.proposed_concept)
        for concept_name, specs in self.added_properties.items():
            concept = evolved.concepts.setdefault(concept_name, ConceptDef(concept_name))
            for spec in specs:
                concept.properties[spec.name] = spec
        for concept_name, relations in self.added_relations.items():
            concept = evolved.concepts.setdefault(concept_name, ConceptDef(concept_name))
            for relation in relations:
                concept.relations[relation.name] = relation
        self.status = "applied"
        return evolved

    def describe(self) -> str:
        lines = [f"TBox proposal {self.id} [{self.status}]",
                 f"  observed gap: {self.observed_gap}"]
        if self.proposed_concept is not None:
            lines.append(f"  proposed concept: {self.proposed_concept.name}")
            for spec in self.proposed_concept.properties.values():
                lines.append(f"    - {spec.name}: {spec.datatype}")
            for relation in self.proposed_concept.relations.values():
                lines.append(f"    -> {relation.name} : {relation.target}")
        for concept_name, specs in self.added_properties.items():
            for spec in specs:
                lines.append(f"  add property {concept_name}.{spec.name}: {spec.datatype}")
        for concept_name, relations in self.added_relations.items():
            for relation in relations:
                lines.append(f"  add relation {concept_name} -{relation.name}-> {relation.target}")
        if self.evidence:
            lines.append(f"  evidence: {', '.join(self.evidence)}")
        if self.rationale:
            lines.append(f"  rationale: {self.rationale}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "observed_gap": self.observed_gap,
            "proposed_concept": self.proposed_concept.to_dict() if self.proposed_concept else None,
            "added_properties": {k: [s.to_dict() for s in v] for k, v in self.added_properties.items()},
            "added_relations": {k: [r.to_dict() for r in v] for k, v in self.added_relations.items()},
            "evidence": list(self.evidence),
            "rationale": self.rationale,
            "status": self.status,
            "created_at": self.created_at,
        }


class OntologyGapDetector:
    """
    Turns representation failures into TBox proposals.

    Two kinds of gap are detected: facts whose concept or property the TBox does
    not declare, and paths a rule (or a human's stated reason) depends on that
    the TBox cannot express.
    """

    def __init__(self, tbox: TBox) -> None:
        self.tbox = tbox

    def from_graph(self, graph: Graph, case_id: str = "") -> List[TBoxEvolutionProposal]:
        proposals: List[TBoxEvolutionProposal] = []
        unknown_concepts: Dict[str, Node] = {}
        undeclared_properties: Dict[str, List[str]] = {}
        for node in graph.walk():
            if not self.tbox.has_concept(node.type):
                unknown_concepts.setdefault(node.type, node)
                continue
            declared = self.tbox.properties_of(node.type)
            for key in node.props:
                if key != "label" and key not in declared:
                    undeclared_properties.setdefault(node.type, [])
                    if key not in undeclared_properties[node.type]:
                        undeclared_properties[node.type].append(key)
        for concept_name, sample in unknown_concepts.items():
            concept = ConceptDef(
                name=concept_name,
                properties={k: PropertySpec(k, _datatype_of(v))
                            for k, v in sample.props.items() if k != "label"},
                description=f"observed in {case_id or sample.id}",
            )
            proposals.append(TBoxEvolutionProposal(
                id=f"tbox-gap-{concept_name}",
                observed_gap=f"Existing ontology cannot represent '{concept_name}'.",
                proposed_concept=concept,
                evidence=(case_id,) if case_id else (),
                rationale="Facts of this concept appeared in a live case.",
            ))
        for concept_name, keys in undeclared_properties.items():
            specs = []
            for key in keys:
                sample_node = next((n for n in graph.walk() if n.type == concept_name and key in n.props), None)
                specs.append(PropertySpec(key, _datatype_of(sample_node.props[key]) if sample_node else "string"))
            proposals.append(TBoxEvolutionProposal(
                id=f"tbox-gap-{concept_name}-props",
                observed_gap=f"'{concept_name}' carries undeclared properties: {', '.join(keys)}.",
                added_properties={concept_name: specs},
                evidence=(case_id,) if case_id else (),
                rationale="Facts used by the decision are not part of the world model.",
            ))
        return proposals

    def from_missing_path(self, concept_name: str, path: str, case_id: str = "",
                          datatype: str = "string") -> TBoxEvolutionProposal:
        """A rule or a human reason needs a path the world model cannot express."""
        return TBoxEvolutionProposal(
            id=f"tbox-gap-{concept_name}-{path.replace('.', '-')}",
            observed_gap=f"Decision depends on '{path}' which '{concept_name}' cannot express.",
            added_properties={concept_name: [PropertySpec(path.split(".")[-1], datatype)]},
            evidence=(case_id,) if case_id else (),
            rationale="Referenced by a rule or by a human decision rationale.",
        )


def _datatype_of(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "string"
