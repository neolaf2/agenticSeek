"""
Semantica: the cognitive-asset substrate.

    TBox  (world model)  - sources.semantica.tbox
    ABox  (facts)        - sources.semantica.abox
    RBox  (rules)        - sources.semantica.rbox
    Skill (procedure)    - sources.semantica.skills
    evolution            - sources.semantica.evolution

The firewall change-request domain built on top of it lives in
``sources.semantica.firewall``.
"""

from sources.semantica.abox import (ABox, ContextSpec, PreferenceCandidate, PreferenceLearner,
                                    PreferenceSignal, SituationABox, SituationBuilder)
from sources.semantica.conditions import All, Always, Any_, Cmp, Condition, Not, parse_condition
from sources.semantica.evolution import (BackTester, BackTestReport, CaseArchive, CaseRecord,
                                         CounterfactualAnalyzer, DecisionDelta, GovernanceGate,
                                         RuleChangeProposal)
from sources.semantica.model import (AssetKind, Binding, Edge, Graph, Node, Provenance, Resolver,
                                     Version)
from sources.semantica.rbox import Action, Decision, RBox, Rule, RuleHit
from sources.semantica.skills import (Branch, Loop, Sequence, SkillChangeProposal, SkillContext,
                                      SkillGraph, SkillRuntime, Step)
from sources.semantica.store import (HttpSemanticaStore, InMemorySemanticaStore, SemanticaStore,
                                     open_store)
from sources.semantica.tbox import (ConceptDef, OntologyGapDetector, PropertySpec, RelationSpec,
                                    TBox, TBoxEvolutionProposal)

__all__ = [
    "ABox", "Action", "All", "Always", "Any_", "AssetKind", "BackTestReport", "BackTester",
    "Binding", "Branch", "CaseArchive", "CaseRecord", "Cmp", "Condition", "ConceptDef",
    "ContextSpec", "CounterfactualAnalyzer", "Decision", "DecisionDelta", "Edge",
    "GovernanceGate", "Graph", "HttpSemanticaStore", "InMemorySemanticaStore", "Loop", "Node",
    "Not", "OntologyGapDetector", "PreferenceCandidate", "PreferenceLearner", "PreferenceSignal",
    "PropertySpec", "Provenance", "RBox", "RelationSpec", "Resolver", "Rule", "RuleChangeProposal",
    "RuleHit", "SemanticaStore", "Sequence", "SituationABox", "SituationBuilder",
    "SkillChangeProposal", "SkillContext", "SkillGraph", "SkillRuntime", "Step", "TBox",
    "TBoxEvolutionProposal", "Version", "open_store", "parse_condition",
]
