"""
Storage contract for Semantica.

Semantica is treated as the substrate that holds ontologies, contextual graphs
and provenance. Two backends implement the same contract:

    InMemorySemanticaStore  - embedded property graph, the default, no server
    HttpSemanticaStore      - thin REST adapter for a Semantica deployment

Everything above this module (TBox / ABox / RBox / Skill / evolution) talks only
to the ``SemanticaStore`` interface, so swapping the backend changes nothing.
"""

from __future__ import annotations

import copy
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sources.semantica.model import AssetKind, Edge, Graph, Node, Version, utc_now


@dataclass
class AssetRecord:
    """A stored version of a cognitive asset."""
    kind: AssetKind
    name: str
    version: str
    payload: Dict[str, Any]
    changelog: str = ""
    parent_version: Optional[str] = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "version": self.version,
            "payload": self.payload,
            "changelog": self.changelog,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AssetRecord":
        return AssetRecord(
            kind=AssetKind(data["kind"]),
            name=data["name"],
            version=data["version"],
            payload=data.get("payload", {}),
            changelog=data.get("changelog", ""),
            parent_version=data.get("parent_version"),
            created_at=data.get("created_at", utc_now()),
        )


class SemanticaStore(ABC):
    """Graph + versioned asset storage."""

    @abstractmethod
    def upsert_node(self, node: Node) -> Node:
        ...

    @abstractmethod
    def upsert_edge(self, edge: Edge) -> Edge:
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        ...

    @abstractmethod
    def match(self, type: Optional[str] = None, ontology: Optional[str] = None, **props: Any) -> List[Node]:
        ...

    @abstractmethod
    def out_edges(self, node_id: str, rel: Optional[str] = None) -> List[Edge]:
        ...

    @abstractmethod
    def in_edges(self, node_id: str, rel: Optional[str] = None) -> List[Edge]:
        ...

    @abstractmethod
    def graph(self, ontology: Optional[str] = None) -> Graph:
        ...

    @abstractmethod
    def put_asset(self, kind: AssetKind, name: str, version: str, payload: Dict[str, Any],
                  changelog: str = "", parent_version: Optional[str] = None) -> AssetRecord:
        ...

    @abstractmethod
    def get_asset(self, kind: AssetKind, name: str, version: Optional[str] = None) -> Optional[AssetRecord]:
        ...

    @abstractmethod
    def list_asset_versions(self, kind: AssetKind, name: str) -> List[AssetRecord]:
        ...

    def load_graph(self, graph: Graph) -> None:
        """Bulk-load a graph, used when seeding ontologies."""
        for node in graph.nodes.values():
            self.upsert_node(node)
        for edge in graph.edges.values():
            self.upsert_edge(edge)


class InMemorySemanticaStore(SemanticaStore):
    """
    Embedded backend.

    Holds every ontology in one property graph namespaced by ``Node.ontology``,
    which is what lets a Situation ABox be assembled across ontologies in a
    single traversal.
    """

    def __init__(self) -> None:
        self._graph = Graph(name="semantica")
        self._assets: Dict[str, List[AssetRecord]] = {}

    # --- graph -----------------------------------------------------------
    def upsert_node(self, node: Node) -> Node:
        return self._graph.add_node(node)

    def upsert_edge(self, edge: Edge) -> Edge:
        return self._graph.add_edge(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._graph.get(node_id)

    def match(self, type: Optional[str] = None, ontology: Optional[str] = None, **props: Any) -> List[Node]:
        return self._graph.match(type=type, ontology=ontology, **props)

    def out_edges(self, node_id: str, rel: Optional[str] = None) -> List[Edge]:
        return self._graph.out_edges(node_id, rel)

    def in_edges(self, node_id: str, rel: Optional[str] = None) -> List[Edge]:
        return self._graph.in_edges(node_id, rel)

    def graph(self, ontology: Optional[str] = None) -> Graph:
        if ontology is None:
            return self._graph
        keep = [n.id for n in self._graph.nodes.values() if n.ontology == ontology]
        return self._graph.subgraph(keep, name=ontology)

    # --- assets ----------------------------------------------------------
    def _key(self, kind: AssetKind, name: str) -> str:
        return f"{kind.value}:{name}"

    def put_asset(self, kind: AssetKind, name: str, version: str, payload: Dict[str, Any],
                  changelog: str = "", parent_version: Optional[str] = None) -> AssetRecord:
        record = AssetRecord(kind=kind, name=name, version=version,
                             payload=copy.deepcopy(payload), changelog=changelog,
                             parent_version=parent_version)
        history = self._assets.setdefault(self._key(kind, name), [])
        history = [r for r in history if r.version != version]
        history.append(record)
        history.sort(key=lambda r: Version.parse(r.version))
        self._assets[self._key(kind, name)] = history
        return record

    def get_asset(self, kind: AssetKind, name: str, version: Optional[str] = None) -> Optional[AssetRecord]:
        history = self._assets.get(self._key(kind, name), [])
        if not history:
            return None
        if version is None:
            return history[-1]
        for record in history:
            if record.version == version:
                return record
        return None

    def list_asset_versions(self, kind: AssetKind, name: str) -> List[AssetRecord]:
        return list(self._assets.get(self._key(kind, name), []))


class HttpSemanticaStore(SemanticaStore):
    """
    REST adapter for a Semantica server.

    The wire format mirrors the in-memory backend one-for-one::

        POST   {base}/nodes                    {node}
        POST   {base}/edges                    {edge}
        GET    {base}/nodes/{id}
        GET    {base}/nodes?type=&ontology=&props={json}
        GET    {base}/edges?src=&rel=  |  ?dst=&rel=
        GET    {base}/graph?ontology=
        POST   {base}/assets/{kind}/{name}     {version, payload, ...}
        GET    {base}/assets/{kind}/{name}[?version=]
        GET    {base}/assets/{kind}/{name}/versions

    The HTTP session is injectable so the request shaping can be tested without
    a live server; this adapter has not been exercised against one.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, session: Any = None,
                 timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if session is not None:
            self.session = session
        else:
            import requests  # imported lazily: the in-memory backend needs no HTTP stack
            self.session = requests.Session()
            if api_key:
                self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    def upsert_node(self, node: Node) -> Node:
        self._request("POST", "/nodes", json=node.to_dict())
        return node

    def upsert_edge(self, edge: Edge) -> Edge:
        self._request("POST", "/edges", json=edge.to_dict())
        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        data = self._request("GET", f"/nodes/{node_id}")
        return Node.from_dict(data) if data else None

    def match(self, type: Optional[str] = None, ontology: Optional[str] = None, **props: Any) -> List[Node]:
        params: Dict[str, Any] = {}
        if type is not None:
            params["type"] = type
        if ontology is not None:
            params["ontology"] = ontology
        if props:
            params["props"] = json.dumps(props, sort_keys=True)
        data = self._request("GET", "/nodes", params=params) or []
        return [Node.from_dict(item) for item in data]

    def out_edges(self, node_id: str, rel: Optional[str] = None) -> List[Edge]:
        params: Dict[str, Any] = {"src": node_id}
        if rel is not None:
            params["rel"] = rel
        data = self._request("GET", "/edges", params=params) or []
        return [Edge.from_dict(item) for item in data]

    def in_edges(self, node_id: str, rel: Optional[str] = None) -> List[Edge]:
        params: Dict[str, Any] = {"dst": node_id}
        if rel is not None:
            params["rel"] = rel
        data = self._request("GET", "/edges", params=params) or []
        return [Edge.from_dict(item) for item in data]

    def graph(self, ontology: Optional[str] = None) -> Graph:
        params = {"ontology": ontology} if ontology else {}
        data = self._request("GET", "/graph", params=params) or {"nodes": [], "edges": []}
        return Graph.from_dict(data)

    def put_asset(self, kind: AssetKind, name: str, version: str, payload: Dict[str, Any],
                  changelog: str = "", parent_version: Optional[str] = None) -> AssetRecord:
        record = AssetRecord(kind=kind, name=name, version=version, payload=payload,
                             changelog=changelog, parent_version=parent_version)
        self._request("POST", f"/assets/{kind.value}/{name}", json=record.to_dict())
        return record

    def get_asset(self, kind: AssetKind, name: str, version: Optional[str] = None) -> Optional[AssetRecord]:
        params = {"version": version} if version else {}
        data = self._request("GET", f"/assets/{kind.value}/{name}", params=params)
        return AssetRecord.from_dict(data) if data else None

    def list_asset_versions(self, kind: AssetKind, name: str) -> List[AssetRecord]:
        data = self._request("GET", f"/assets/{kind.value}/{name}/versions") or []
        return [AssetRecord.from_dict(item) for item in data]


def open_store(url: Optional[str] = None, api_key: Optional[str] = None) -> SemanticaStore:
    """
    Open the configured store.

    Uses ``SEMANTICA_URL`` / ``SEMANTICA_API_KEY`` when present and falls back to
    the embedded backend, so the agent runs offline out of the box.
    """
    url = url or os.getenv("SEMANTICA_URL", "")
    if url:
        return HttpSemanticaStore(url, api_key or os.getenv("SEMANTICA_API_KEY"))
    return InMemorySemanticaStore()
