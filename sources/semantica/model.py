"""
Domain model for the Semantica cognitive-asset substrate.

Everything an agent knows is expressed as a property graph (nodes + edges) whose
elements carry provenance, and which is grouped into *cognitive assets*:

    TBox  -> what concepts may exist          (world model / schema)
    ABox  -> what is currently true           (facts, situation, preferences)
    RBox  -> what must happen when            (rules, exceptions)
    Skill -> how the work is carried out      (sequence / condition / loop)

Each asset is versioned, so evolution is auditable: an asset is never mutated in
place, a new version is derived from the previous one with a changelog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


def utc_now() -> str:
    """Timestamp used for every provenance record."""
    return datetime.now(timezone.utc).isoformat()


class AssetKind(str, Enum):
    """The four kinds of cognitive asset that can be learned independently."""
    TBOX = "tbox"
    ABOX = "abox"
    RBOX = "rbox"
    SKILL = "skill"


@dataclass(frozen=True, order=True)
class Version:
    """A two-part version (major.minor) for a cognitive asset."""
    major: int = 1
    minor: int = 0

    @staticmethod
    def parse(text: str) -> "Version":
        match = re.fullmatch(r"v?(\d+)\.(\d+)", str(text).strip())
        if match is None:
            raise ValueError(f"invalid version: {text!r}")
        return Version(int(match.group(1)), int(match.group(2)))

    def bump_minor(self) -> "Version":
        return Version(self.major, self.minor + 1)

    def bump_major(self) -> "Version":
        return Version(self.major + 1, 0)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class Provenance:
    """
    Where a fact came from and how much we trust it.

    Attributes:
        source: The cognitive asset or system that asserted the element.
        derived_from: Identifiers of the evidence this was derived from.
        confidence: Belief in the element, 0.0 - 1.0.
        asset_version: Version of the asset at assertion time.
        recorded_at: ISO timestamp of the assertion.
    """
    source: str = "unknown"
    derived_from: Tuple[str, ...] = ()
    confidence: float = 1.0
    asset_version: str = ""
    recorded_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "derived_from": list(self.derived_from),
            "confidence": self.confidence,
            "asset_version": self.asset_version,
            "recorded_at": self.recorded_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Provenance":
        return Provenance(
            source=data.get("source", "unknown"),
            derived_from=tuple(data.get("derived_from", ())),
            confidence=float(data.get("confidence", 1.0)),
            asset_version=data.get("asset_version", ""),
            recorded_at=data.get("recorded_at", utc_now()),
        )


@dataclass
class Node:
    """An entity in the property graph."""
    id: str
    type: str
    props: Dict[str, Any] = field(default_factory=dict)
    ontology: str = "default"
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def label(self) -> str:
        return str(self.props.get("label", self.id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "props": dict(self.props),
            "ontology": self.ontology,
            "provenance": self.provenance.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Node":
        return Node(
            id=data["id"],
            type=data["type"],
            props=dict(data.get("props", {})),
            ontology=data.get("ontology", "default"),
            provenance=Provenance.from_dict(data.get("provenance", {})),
        )


@dataclass
class Edge:
    """A typed relation between two nodes."""
    src: str
    rel: str
    dst: str
    props: Dict[str, Any] = field(default_factory=dict)
    ontology: str = "default"
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def id(self) -> str:
        return f"{self.src}|{self.rel}|{self.dst}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "rel": self.rel,
            "dst": self.dst,
            "props": dict(self.props),
            "ontology": self.ontology,
            "provenance": self.provenance.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Edge":
        return Edge(
            src=data["src"],
            rel=data["rel"],
            dst=data["dst"],
            props=dict(data.get("props", {})),
            ontology=data.get("ontology", "default"),
            provenance=Provenance.from_dict(data.get("provenance", {})),
        )


@dataclass(frozen=True)
class Binding:
    """
    One value produced by resolving a path against a graph.

    Keeping the originating node and edge alongside the value is what makes an
    evidence graph possible: every rule hit can point back at the exact fact.
    """
    path: str
    value: Any
    node_id: str
    edge: Optional[Edge] = None

    @property
    def ontology(self) -> str:
        return self.edge.ontology if self.edge is not None else ""


class Resolver:
    """
    Anything a condition can be evaluated against.

    Both a Situation ABox and a running skill context are resolvers, which is why
    the same condition language serves the RBox and skill branching.
    """

    def resolve(self, path: str) -> List[Binding]:
        raise NotImplementedError


@dataclass
class Graph(Resolver):
    """A property graph: the storage shape shared by every cognitive asset."""
    name: str = "graph"
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)
    root_id: Optional[str] = None

    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        self.edges[edge.id] = edge
        return edge

    def get(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def match(self, type: Optional[str] = None, ontology: Optional[str] = None, **props: Any) -> List[Node]:
        found = []
        for node in self.nodes.values():
            if type is not None and node.type != type:
                continue
            if ontology is not None and node.ontology != ontology:
                continue
            if any(node.props.get(key) != value for key, value in props.items()):
                continue
            found.append(node)
        return found

    def out_edges(self, node_id: str, rel: Optional[str] = None) -> List[Edge]:
        return [e for e in self.edges.values()
                if e.src == node_id and (rel is None or e.rel == rel)]

    def in_edges(self, node_id: str, rel: Optional[str] = None) -> List[Edge]:
        return [e for e in self.edges.values()
                if e.dst == node_id and (rel is None or e.rel == rel)]

    def merge(self, other: "Graph") -> "Graph":
        """Fold another graph into this one; existing elements are kept."""
        for node in other.nodes.values():
            self.nodes.setdefault(node.id, node)
        for edge in other.edges.values():
            self.edges.setdefault(edge.id, edge)
        return self

    def subgraph(self, node_ids: Sequence[str], name: Optional[str] = None) -> "Graph":
        keep = set(node_ids)
        sub = Graph(name=name or f"{self.name}-sub")
        for node_id in keep:
            node = self.nodes.get(node_id)
            if node is not None:
                sub.add_node(node)
        for edge in self.edges.values():
            if edge.src in keep and edge.dst in keep:
                sub.add_edge(edge)
        return sub

    def resolve(self, path: str, root_id: Optional[str] = None) -> List[Binding]:
        """
        Walk a dotted path such as ``destination.networkZone`` from a root node.

        Each segment is followed as a relation when one exists and read as a
        property otherwise; the special segment ``#type`` yields the node type.
        Missing segments simply produce no bindings - absence is expressed by the
        ``missing`` operator rather than by an exception.
        """
        start = root_id or self.root_id
        if start is None or start not in self.nodes:
            return []
        segments = [s for s in path.split(".") if s]
        current: List[str] = [start]
        collected: List[Binding] = []
        for index, segment in enumerate(segments):
            is_last = index == len(segments) - 1
            following: List[str] = []
            for node_id in current:
                node = self.nodes.get(node_id)
                if node is None:
                    continue
                if segment == "#type":
                    if is_last:
                        collected.append(Binding(path, node.type, node_id))
                    continue
                for edge in self.out_edges(node_id, segment):
                    following.append(edge.dst)
                    if is_last:
                        target = self.nodes.get(edge.dst)
                        value = target.label if target is not None else edge.dst
                        collected.append(Binding(path, value, edge.dst, edge))
                if is_last and segment in node.props:
                    collected.append(Binding(path, node.props[segment], node_id))
            current = following
            if not current and not is_last:
                break
        return collected

    def walk(self) -> Iterator[Node]:
        return iter(self.nodes.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "root_id": self.root_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Graph":
        graph = Graph(name=data.get("name", "graph"), root_id=data.get("root_id"))
        for node in data.get("nodes", []):
            graph.add_node(Node.from_dict(node))
        for edge in data.get("edges", []):
            graph.add_edge(Edge.from_dict(edge))
        return graph

    def __len__(self) -> int:
        return len(self.nodes)
