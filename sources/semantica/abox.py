"""
ABox: what is currently true.

The ABox an agent reasons over is not one big permanent graph. It is a *Situation
ABox*: a working subgraph assembled at run time by pulling facts out of several
ontologies at once (organisational, infrastructure, personal, task) around the
entity that triggered the work.

    Personal ABox ----\\
    Org ABox ----------\\
    Infrastructure -----> Situation ABox
    Task ABox ---------//
    Ticket ABox ------//

Preferences are handled here too, because a stable preference is a fact that was
learned rather than loaded: repeated signals become a Personal ABox assertion
with a confidence, a scope and the sessions it came from.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sources.semantica.model import (AssetKind, Binding, Graph, Node, Provenance, Resolver,
                                     utc_now)
from sources.semantica.store import SemanticaStore
from sources.semantica.tbox import PropertySpec, TBox, TBoxEvolutionProposal


@dataclass
class ContextSpec:
    """
    How wide to open the lens when building a situation.

    Attributes:
        hops: Traversal depth from the root entity.
        follow: Relation whitelist; empty means follow every relation.
        ontologies: Ontology whitelist; empty means every ontology.
        include_incoming: Also pull facts that point *at* the entities reached.
    """
    hops: int = 3
    follow: Tuple[str, ...] = ()
    ontologies: Tuple[str, ...] = ()
    include_incoming: bool = False


@dataclass
class SituationABox(Resolver):
    """
    The working subgraph for one case.

    It is a Resolver, so rules and skill branches read it directly by path.
    """
    case_id: str
    graph: Graph
    root_id: str
    sources: Dict[str, int] = field(default_factory=dict)
    built_at: str = field(default_factory=utc_now)

    def resolve(self, path: str) -> List[Binding]:
        return self.graph.resolve(path, self.root_id)

    def value(self, path: str, default: Any = None) -> Any:
        bindings = self.resolve(path)
        return bindings[0].value if bindings else default

    def values(self, path: str) -> List[Any]:
        return [b.value for b in self.resolve(path)]

    @property
    def root(self) -> Optional[Node]:
        return self.graph.get(self.root_id)

    def facts(self) -> List[str]:
        """Flatten the situation into readable ``subject.predicate = value`` lines."""
        lines: List[str] = []
        for node in sorted(self.graph.walk(), key=lambda n: n.id):
            for key, value in sorted(node.props.items()):
                if key == "label":
                    continue
                lines.append(f"{node.id}.{key} = {value}   [{node.ontology}]")
        for edge in sorted(self.graph.edges.values(), key=lambda e: e.id):
            lines.append(f"{edge.src} -{edge.rel}-> {edge.dst}   [{edge.ontology}]")
        return lines

    def explain(self, node_id: str) -> str:
        node = self.graph.get(node_id)
        if node is None:
            return f"{node_id}: not part of this situation"
        provenance = node.provenance
        return (f"{node_id} ({node.type}) from ontology '{node.ontology}', "
                f"source={provenance.source}, confidence={provenance.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {"case_id": self.case_id, "root_id": self.root_id, "sources": dict(self.sources),
                "built_at": self.built_at, "graph": self.graph.to_dict()}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SituationABox":
        return SituationABox(case_id=data["case_id"], graph=Graph.from_dict(data["graph"]),
                             root_id=data["root_id"], sources=dict(data.get("sources", {})),
                             built_at=data.get("built_at", utc_now()))


class SituationBuilder:
    """Assembles a Situation ABox by traversing the store from a root entity."""

    def __init__(self, store: SemanticaStore, spec: Optional[ContextSpec] = None) -> None:
        self.store = store
        self.spec = spec or ContextSpec()

    def build(self, root_id: str, case_id: Optional[str] = None,
              extra_roots: Sequence[str] = ()) -> SituationABox:
        """
        Args:
            root_id: Entity the work is about, typically the ticket.
            case_id: Identifier used in provenance and case archives.
            extra_roots: Additional entry points, e.g. the task ontology node
                describing which checks this kind of request requires.
        """
        graph = Graph(name=f"situation:{case_id or root_id}", root_id=root_id)
        counts: Counter = Counter()
        queue: deque = deque((node_id, 0) for node_id in [root_id, *extra_roots])
        seen: set = set()
        while queue:
            node_id, depth = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.store.get_node(node_id)
            if node is None:
                continue
            if self.spec.ontologies and node.ontology not in self.spec.ontologies and node_id != root_id:
                continue
            graph.add_node(node)
            counts[node.ontology] += 1
            if depth >= self.spec.hops:
                continue
            edges = list(self.store.out_edges(node_id))
            if self.spec.include_incoming:
                edges.extend(self.store.in_edges(node_id))
            for edge in edges:
                if self.spec.follow and edge.rel not in self.spec.follow:
                    continue
                graph.add_edge(edge)
                queue.append((edge.dst if edge.src == node_id else edge.src, depth + 1))
        # Drop dangling edges so the situation stays self-contained and auditable.
        for edge_id, edge in list(graph.edges.items()):
            if edge.src not in graph.nodes or edge.dst not in graph.nodes:
                del graph.edges[edge_id]
        return SituationABox(case_id=case_id or root_id, graph=graph, root_id=root_id,
                             sources=dict(counts))


@dataclass
class ABox:
    """A named, versioned set of facts belonging to one ontology."""
    name: str
    graph: Graph
    version: str = "1.0"

    def save(self, store: SemanticaStore, changelog: str = "") -> None:
        store.load_graph(self.graph)
        store.put_asset(AssetKind.ABOX, self.name, self.version, self.graph.to_dict(),
                        changelog=changelog)


@dataclass(frozen=True)
class PreferenceSignal:
    """One observation that a person prefers something, in some scope."""
    subject: str
    predicate: str
    value: Any
    scope: str = "global"
    source: str = ""
    observed_at: str = field(default_factory=utc_now)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.scope)


@dataclass
class PreferenceCandidate:
    """A preference that is stable enough to be worth asserting."""
    subject: str
    predicate: str
    value: Any
    scope: str
    confidence: float
    sources: Tuple[str, ...]
    last_confirmed: str
    contradictions: int = 0

    def describe(self) -> str:
        return (f"{self.subject} {self.predicate} {self.value} "
                f"(confidence={self.confidence:.2f}, scope={self.scope}, "
                f"sources={list(self.sources)})")


class PreferenceLearner:
    """
    Episodic memory -> Personal ABox.

    The first time someone says "for production firewall changes I would rather
    have one more human review" it is just session memory. Repetition turns it
    into a fact with a confidence. If the personal ontology has no concept for
    it, that is a *different* kind of learning, and a TBox proposal is emitted
    alongside the fact.
    """

    def __init__(self, threshold: float = 0.8, ontology: str = "personal") -> None:
        self.threshold = threshold
        self.ontology = ontology
        self._signals: Dict[Tuple[str, str, str], List[PreferenceSignal]] = {}

    def observe(self, signal: PreferenceSignal) -> None:
        self._signals.setdefault(signal.key, []).append(signal)

    def candidates(self) -> List[PreferenceCandidate]:
        results: List[PreferenceCandidate] = []
        for (subject, predicate, scope), signals in self._signals.items():
            counts = Counter(str(s.value) for s in signals)
            top_value, top_count = counts.most_common(1)[0]
            contradictions = len(signals) - top_count
            support = 1.0 - 0.5 ** top_count
            agreement = top_count / len(signals)
            supporting = [s for s in signals if str(s.value) == top_value]
            results.append(PreferenceCandidate(
                subject=subject,
                predicate=predicate,
                value=supporting[-1].value,
                scope=scope,
                confidence=round(support * agreement, 4),
                sources=tuple(s.source for s in supporting if s.source),
                last_confirmed=supporting[-1].observed_at,
                contradictions=contradictions,
            ))
        return sorted(results, key=lambda c: c.confidence, reverse=True)

    def promote(self, store: SemanticaStore, tbox: Optional[TBox] = None,
                person_concept: str = "Person") -> Tuple[List[PreferenceCandidate], List[TBoxEvolutionProposal]]:
        """
        Write confident preferences into the Personal ABox.

        Returns:
            The promoted candidates and any TBox proposals required because the
            personal ontology cannot yet express the preference.
        """
        promoted: List[PreferenceCandidate] = []
        proposals: List[TBoxEvolutionProposal] = []
        for candidate in self.candidates():
            if candidate.confidence < self.threshold:
                continue
            if tbox is not None and candidate.predicate not in tbox.properties_of(person_concept):
                proposals.append(TBoxEvolutionProposal(
                    id=f"tbox-personal-{candidate.predicate}",
                    observed_gap=(f"Personal ontology has no '{candidate.predicate}' on "
                                  f"'{person_concept}'."),
                    added_properties={person_concept: [PropertySpec(candidate.predicate, "string")]},
                    evidence=candidate.sources,
                    rationale="A stable user preference cannot be asserted without it.",
                ))
            node = store.get_node(candidate.subject)
            if node is None:
                node = Node(id=candidate.subject, type=person_concept, ontology=self.ontology)
            node.props[candidate.predicate] = candidate.value
            # Scope stays out of the props so the fact remains TBox-valid; it is
            # carried by provenance, next to the sessions that produced it.
            node.provenance = Provenance(
                source=f"preference-learner:{candidate.scope}",
                derived_from=candidate.sources,
                confidence=candidate.confidence,
            )
            store.upsert_node(node)
            promoted.append(candidate)
        return promoted, proposals
