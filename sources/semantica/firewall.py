"""
The firewall change-request domain, expressed as cognitive assets.

This module seeds the four assets (world model, facts, rules, skill), ingests a
ticket into a Situation ABox and runs the review skill over it. The pipeline is
the object the agent talks to; everything it does is replayable without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sources.semantica.abox import (ContextSpec, PreferenceLearner, PreferenceSignal,
                                    SituationABox, SituationBuilder)
from sources.semantica.conditions import All, Cmp
from sources.semantica.evolution import (CaseArchive, CaseRecord, CounterfactualAnalyzer,
                                         DecisionDelta, GovernanceGate, RuleChangeProposal,
                                         record_proposal)
from sources.semantica.model import Edge, Node, Provenance, utc_now
from sources.semantica.rbox import Action, Decision, RBox, Rule
from sources.semantica.skills import (Branch, Loop, Sequence as SkillSequence,
                                      SkillChangeProposal, SkillContext, SkillGraph,
                                      SkillRuntime, Step)
from sources.semantica.store import InMemorySemanticaStore, SemanticaStore
from sources.semantica.tbox import (ConceptDef, OntologyGapDetector, PropertySpec, RelationSpec,
                                    TBox, TBoxEvolutionProposal)

TBOX_NAME = "firewall-world-model"
RBOX_NAME = "firewall-policy"
SKILL_ID = "EvaluateFirewallRequestSkill"

ONTOLOGY_ORG = "organizational"
ONTOLOGY_INFRA = "infrastructure"
ONTOLOGY_PERSONAL = "personal"
ONTOLOGY_TASK = "task"
ONTOLOGY_TICKET = "ticket"


# --------------------------------------------------------------------------
# TBox v1.0
# --------------------------------------------------------------------------
def build_tbox() -> TBox:
    """The initial world model: deliberately incomplete, so it can be grown."""
    return TBox(TBOX_NAME, "1.0", [
        ConceptDef("Thing", properties={"label": PropertySpec("label")}),
        ConceptDef("Team", parents=("Thing",),
                   properties={"costCenter": PropertySpec("costCenter")}),
        ConceptDef("Person", parents=("Thing",), properties={
            "role": PropertySpec("role"),
            "approvalAuthority": PropertySpec("approvalAuthority", allowed=("Level1", "Level2", "Level3")),
            "department": PropertySpec("department"),
        }, relations={"memberOf": RelationSpec("memberOf", "Team")},
           description="A requester, an owner or a reviewer."),
        ConceptDef("Application", parents=("Thing",), properties={
            "environment": PropertySpec("environment", allowed=("Development", "Staging", "Production")),
            "businessCriticality": PropertySpec("businessCriticality", allowed=("Tier1", "Tier2", "Tier3")),
            "lifecycleStage": PropertySpec("lifecycleStage"),
        }, relations={"owner": RelationSpec("owner", "Team")}),
        ConceptDef("NetworkZone", parents=("Thing",), properties={
            "trustLevel": PropertySpec("trustLevel", allowed=("trusted", "restricted", "untrusted")),
        }),
        ConceptDef("NetworkSegment", parents=("Thing",), properties={"cidr": PropertySpec("cidr")},
                   relations={"networkZone": RelationSpec("networkZone", "NetworkZone")}),
        ConceptDef("Server", parents=("Thing",), properties={
            "dataClassification": PropertySpec("dataClassification",
                                               allowed=("Public", "Internal", "Confidential", "Restricted")),
            "environment": PropertySpec("environment", allowed=("Development", "Staging", "Production")),
        }, relations={"networkZone": RelationSpec("networkZone", "NetworkZone", required=True),
                      "hosts": RelationSpec("hosts", "Application", functional=False)}),
        ConceptDef("Port", parents=("Thing",), properties={
            "number": PropertySpec("number", "int"),
            "protocol": PropertySpec("protocol"),
            "riskLevel": PropertySpec("riskLevel", allowed=("low", "medium", "high", "unknown")),
            "service": PropertySpec("service"),
        }),
        ConceptDef("Check", parents=("Thing",), properties={"mandatory": PropertySpec("mandatory", "bool")}),
        ConceptDef("TaskDefinition", parents=("Thing",),
                   relations={"requires": RelationSpec("requires", "Check", functional=False)}),
        ConceptDef("FirewallChangeRequest", parents=("Thing",), properties={
            "justification": PropertySpec("justification"),
            "duration": PropertySpec("duration"),
            "openedAt": PropertySpec("openedAt"),
        }, relations={
            "requester": RelationSpec("requester", "Person", required=True),
            "application": RelationSpec("application", "Application"),
            "source": RelationSpec("source", "NetworkSegment", required=True),
            "destination": RelationSpec("destination", "Server", required=True),
            "port": RelationSpec("port", "Port", required=True),
            "reviewedBy": RelationSpec("reviewedBy", "Person"),
        }, description="A request to open a network path."),
    ])


# --------------------------------------------------------------------------
# ABox seeds: four ontologies that the Situation ABox is assembled from
# --------------------------------------------------------------------------
def seed_ontologies(store: SemanticaStore) -> None:
    """Populate the organisational, infrastructure, personal and task ontologies."""
    def node(node_id: str, type_: str, ontology: str, **props: Any) -> None:
        store.upsert_node(Node(id=node_id, type=type_, ontology=ontology, props=props,
                               provenance=Provenance(source=f"seed:{ontology}")))

    def edge(src: str, rel: str, dst: str, ontology: str) -> None:
        store.upsert_edge(Edge(src, rel, dst, ontology=ontology,
                               provenance=Provenance(source=f"seed:{ontology}")))

    # organisational
    node("payments-team", "Team", ONTOLOGY_ORG, label="Payments Team", costCenter="CC-4410")
    node("platform-team", "Team", ONTOLOGY_ORG, label="Platform Team", costCenter="CC-2201")
    node("PaymentApp", "Application", ONTOLOGY_ORG, label="PaymentApp", environment="Production",
         businessCriticality="Tier1", lifecycleStage="Operating")
    node("ReportingApp", "Application", ONTOLOGY_ORG, label="ReportingApp", environment="Staging",
         businessCriticality="Tier3", lifecycleStage="Operating")
    edge("PaymentApp", "owner", "payments-team", ONTOLOGY_ORG)
    edge("ReportingApp", "owner", "platform-team", ONTOLOGY_ORG)
    node("alice", "Person", ONTOLOGY_ORG, label="Alice", role="ApplicationOwner",
         approvalAuthority="Level2", department="Payments")
    node("bob", "Person", ONTOLOGY_ORG, label="Bob", role="Developer",
         approvalAuthority="Level1", department="Payments")
    edge("alice", "memberOf", "payments-team", ONTOLOGY_ORG)
    edge("bob", "memberOf", "payments-team", ONTOLOGY_ORG)

    # infrastructure
    node("PCI-Zone", "NetworkZone", ONTOLOGY_INFRA, label="PCI-Zone", trustLevel="restricted")
    node("Internal-Zone", "NetworkZone", ONTOLOGY_INFRA, label="Internal-Zone", trustLevel="trusted")
    node("DMZ", "NetworkZone", ONTOLOGY_INFRA, label="DMZ", trustLevel="untrusted")
    node("DB-PROD-07", "Server", ONTOLOGY_INFRA, label="DB-PROD-07",
         dataClassification="Restricted", environment="Production")
    node("APP-PROD-11", "Server", ONTOLOGY_INFRA, label="APP-PROD-11",
         dataClassification="Confidential", environment="Production")
    node("RPT-STG-02", "Server", ONTOLOGY_INFRA, label="RPT-STG-02",
         dataClassification="Internal", environment="Staging")
    edge("DB-PROD-07", "networkZone", "PCI-Zone", ONTOLOGY_INFRA)
    edge("APP-PROD-11", "networkZone", "PCI-Zone", ONTOLOGY_INFRA)
    edge("RPT-STG-02", "networkZone", "Internal-Zone", ONTOLOGY_INFRA)
    edge("DB-PROD-07", "hosts", "PaymentApp", ONTOLOGY_INFRA)
    edge("RPT-STG-02", "hosts", "ReportingApp", ONTOLOGY_INFRA)
    for cidr, zone in (("10.23.1.0/24", "Internal-Zone"), ("10.90.4.0/24", "PCI-Zone"),
                       ("172.16.8.0/24", "DMZ")):
        node(cidr, "NetworkSegment", ONTOLOGY_INFRA, label=cidr, cidr=cidr)
        edge(cidr, "networkZone", zone, ONTOLOGY_INFRA)
    for number, service, protocol, risk in ((5432, "postgresql", "tcp", "medium"),
                                            (443, "https", "tcp", "low"),
                                            (22, "ssh", "tcp", "high"),
                                            (3389, "rdp", "tcp", "high")):
        node(f"port:{number}", "Port", ONTOLOGY_INFRA, label=f"{protocol}/{number}", number=number,
             protocol=protocol, service=service, riskLevel=risk)

    # personal (the reviewer this agent works for; preferences are learned, not seeded)
    node("richard", "Person", ONTOLOGY_PERSONAL, label="Richard", role="SecurityReviewer",
         approvalAuthority="Level3", department="Security")

    # task
    node("task:FirewallChangeRequest", "TaskDefinition", ONTOLOGY_TASK,
         label="FirewallChangeRequest")
    for check, mandatory in (("check:ownership", True), ("check:network-path", True),
                             ("check:security-policy", True)):
        node(check, "Check", ONTOLOGY_TASK, label=check.split(":")[1], mandatory=mandatory)
        edge("task:FirewallChangeRequest", "requires", check, ONTOLOGY_TASK)


# --------------------------------------------------------------------------
# RBox v1.0
# --------------------------------------------------------------------------
def build_rbox() -> RBox:
    """
    The starting policy.

    R102 is the rule the vendor case will later prove incomplete; it is written
    exactly as a policy author would write it today.
    """
    scope = Cmp("#type", "eq", "FirewallChangeRequest")
    return RBox(RBOX_NAME, "1.0", [
        Rule(id="R101", version="1.0", priority=10, scope=scope,
             when=All([Cmp("destination.environment", "eq", "Production"),
                       Cmp("requester.approvalAuthority", "in", ["Level1"])]),
             action=Action.MANUAL_REVIEW,
             rationale="Production change raised by a requester without owner-level authority.",
             source="policy:change-management-4.2"),
        Rule(id="R102", version="1.0", priority=20, scope=scope,
             when=All([Cmp("destination.networkZone", "eq", "PCI-Zone"),
                       Cmp("source.networkZone", "ne", "PCI-Zone")]),
             action=Action.SECURITY_REVIEW,
             rationale="Traffic entering the PCI zone from outside it needs security review.",
             source="policy:pci-dss-1.2"),
        Rule(id="R103", version="1.0", priority=30, scope=scope,
             when=All([Cmp("port.riskLevel", "eq", "high")]),
             action=Action.SECURITY_REVIEW,
             rationale="Administrative protocols are never opened without security review.",
             source="policy:pci-dss-2.3"),
        Rule(id="R104", version="1.0", priority=40, scope=scope,
             when=All([Cmp("destination.dataClassification", "eq", "Restricted"),
                       Cmp("source.networkZone.trustLevel", "eq", "untrusted")]),
             action=Action.REJECT,
             rationale="Restricted data is never reachable from an untrusted zone.",
             source="policy:data-protection-3.1"),
        Rule(id="R105", version="1.0", priority=50, scope=scope,
             when=All([Cmp("destination.environment", "eq", "Production"),
                       Cmp("reviewedBy.prefersApprovalMode", "eq", "Conservative")]),
             action=Action.MANUAL_REVIEW,
             rationale="The reviewer prefers a human pass on production firewall changes.",
             source="personal-preference"),
    ], default_action=Action.AUTO_APPROVE)


# --------------------------------------------------------------------------
# Skill v1.0
# --------------------------------------------------------------------------
def build_skill() -> SkillGraph:
    """The 13-step review procedure as a computational graph."""
    root = SkillSequence("review", [
        Step("s1_validate_input", "validate_input", out_key="validation"),
        Step("s2_resolve_entities", "resolve_entities", out_key="resolution"),
        Step("s3_construct_situation", "construct_situation", out_key="situation_summary"),
        Step("s4_retrieve_rbox", "retrieve_rbox", out_key="applicable_rules"),
        Step("s5_ownership_check", "run_ownership_check", out_key="ownership"),
        Step("s6_network_check", "run_network_check", out_key="network"),
        Step("s7_security_policy_check", "run_security_policy_check", out_key="security"),
        Loop("s8_retrieve_missing_evidence", over="missing_paths", var="missing_path",
             body=Step("s8a_lookup", "retrieve_missing_evidence")),
        Step("s9_evaluate_rules", "evaluate_rules", out_key="decision"),
        Step("s10_generate_recommendation", "generate_recommendation", out_key="recommendation"),
        Step("s11_generate_evidence_graph", "generate_evidence_graph", out_key="evidence_graph"),
        Step("s12_request_human_decision", "request_human_decision", out_key="human_request"),
        Step("s13_learn_from_outcome", "learn_from_outcome", out_key="learning"),
    ])
    return SkillGraph(SKILL_ID, "Evaluate firewall change request", root, "1.0",
                      description="Sequence / condition / loop over the review operations.")


def vendor_risk_branch() -> Branch:
    """
    The patch the vendor cases justify.

    It is a *skill* change, not a rule change: the procedure was missing a step,
    so an external-vendor request now gets a risk assessment before the security
    policy check runs.
    """
    return Branch(
        id="s5b_vendor_branch",
        condition=Cmp("requester.organizationType", "eq", "ExternalVendor"),
        then=Step("s5c_vendor_risk_assessment", "vendor_risk_assessment", out_key="vendor_risk"),
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class ReviewResult:
    """Everything one review produced."""
    case_id: str
    decision: Decision
    situation: SituationABox
    trace: List[str]
    checks: Dict[str, Any] = field(default_factory=dict)
    tbox_violations: Tuple[str, ...] = ()
    tbox_proposals: Tuple[TBoxEvolutionProposal, ...] = ()
    skill_version: str = ""

    @property
    def action(self) -> Action:
        return self.decision.action

    def report(self) -> str:
        lines = [f"# Firewall CR review - {self.case_id}",
                 "",
                 f"**Recommendation: {self.decision.action.value}**",
                 "",
                 "## Why",
                 self.decision.explain(),
                 "",
                 "## Checks"]
        for name, outcome in self.checks.items():
            lines.append(f"- {name}: {outcome}")
        lines += ["", "## Situation ABox",
                  f"assembled from {self.situation.sources} "
                  f"({len(self.situation.graph)} entities)"]
        lines.extend(f"- {fact}" for fact in self.situation.facts()[:24])
        lines += ["", "## Evidence graph",
                  f"{len(self.decision.evidence_graph)} nodes / "
                  f"{len(self.decision.evidence_graph.edges)} edges rooted at "
                  f"decision:{self.case_id}"]
        if self.tbox_violations:
            lines += ["", "## World-model violations"]
            lines.extend(f"- {violation}" for violation in self.tbox_violations)
        if self.tbox_proposals:
            lines += ["", "## Proposed TBox evolution"]
            lines.extend(proposal.describe() for proposal in self.tbox_proposals)
        lines += ["", "## Skill trace", f"skill {SKILL_ID} v{self.skill_version}"]
        lines.extend(f"- {entry}" for entry in self.trace)
        lines += ["", "Reply with a decision to close the loop, e.g. "
                  f'`{{"case_id": "{self.case_id}", "human_decision": "SECURITY_REVIEW", '
                  '"reason": "..."}`']
        return "\n".join(lines)


@dataclass
class LearningOutcome:
    """The deltas a closed case produced."""
    delta: DecisionDelta
    rule_proposals: Tuple[RuleChangeProposal, ...] = ()
    tbox_proposals: Tuple[TBoxEvolutionProposal, ...] = ()
    skill_proposals: Tuple[SkillChangeProposal, ...] = ()
    preference_promotions: Tuple[str, ...] = ()

    @property
    def has_proposals(self) -> bool:
        return bool(self.rule_proposals or self.tbox_proposals or self.skill_proposals)

    def report(self) -> str:
        lines = [f"# Learning from {self.delta.case_id}", "", self.delta.describe(), ""]
        if not self.has_proposals:
            lines.append("No asset change is needed: the rules already explain this decision.")
            return "\n".join(lines)
        if self.rule_proposals:
            lines += ["## ΔRBox (rule change proposals)"]
            lines.extend(proposal.describe() for proposal in self.rule_proposals)
            lines.append("")
        if self.tbox_proposals:
            lines += ["## ΔTBox (world-model proposals)"]
            lines.extend(proposal.describe() for proposal in self.tbox_proposals)
            lines.append("")
        if self.skill_proposals:
            lines += ["## ΔSkill (procedure proposals)"]
            lines.extend(proposal.describe() for proposal in self.skill_proposals)
            lines.append("")
        if self.preference_promotions:
            lines += ["## ΔABox (learned facts)"]
            lines.extend(f"- {entry}" for entry in self.preference_promotions)
            lines.append("")
        lines.append("Nothing is live yet. Approve a proposal to derive the next asset version.")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class FirewallReviewPipeline:
    """
    Wires the four cognitive assets into one reviewable, learnable loop.

    Args:
        store: Semantica backend; an embedded one is created when omitted.
        operator: The person the agent reviews on behalf of (personal ontology).
        seed: Seed the firewall ontologies and assets on construction.
    """

    def __init__(self, store: Optional[SemanticaStore] = None, operator: str = "richard",
                 seed: bool = True, archive: Optional[CaseArchive] = None) -> None:
        self.store = store or InMemorySemanticaStore()
        self.operator = operator
        self.archive = archive or CaseArchive()
        self.tbox = build_tbox()
        self.rbox = build_rbox()
        self.skill = build_skill()
        self.preferences = PreferenceLearner()
        self.gate = GovernanceGate(self.archive, self.store)
        self.open_cases: Dict[str, ReviewResult] = {}
        self.proposals: Dict[str, Any] = {}
        self.runtime = SkillRuntime(self._operations())
        if seed:
            seed_ontologies(self.store)
            self.tbox.save(self.store, changelog="initial world model")
            self.rbox.save(self.store, changelog="initial policy")
            self.skill.save(self.store, changelog="initial review procedure")

    # --- ingestion -------------------------------------------------------
    def ingest_ticket(self, ticket: Dict[str, Any]) -> str:
        """
        Write the ticket into the ticket ontology and return its node id.

        Entity references may be plain identifiers or dictionaries; a dictionary
        also asserts attributes, which is how facts the world model does not yet
        declare (an external vendor, say) enter the graph.
        """
        case_id = str(ticket.get("id") or ticket.get("case_id") or f"FW-{utc_now()}")
        props = {key: ticket[key] for key in ("justification", "duration", "openedAt")
                 if ticket.get(key) is not None}
        props["label"] = case_id
        self.store.upsert_node(Node(id=case_id, type="FirewallChangeRequest",
                                    ontology=ONTOLOGY_TICKET, props=props,
                                    provenance=Provenance(source=f"ticket:{case_id}")))
        references = {
            "requester": ("Person", ONTOLOGY_ORG),
            "application": ("Application", ONTOLOGY_ORG),
            "source": ("NetworkSegment", ONTOLOGY_INFRA),
            "destination": ("Server", ONTOLOGY_INFRA),
        }
        for relation, (concept, ontology) in references.items():
            value = ticket.get(relation)
            if value is None:
                continue
            target_id = self._upsert_reference(value, concept, ontology, case_id)
            self.store.upsert_edge(Edge(case_id, relation, target_id, ontology=ONTOLOGY_TICKET,
                                        provenance=Provenance(source=f"ticket:{case_id}")))
        port = ticket.get("port")
        if port is not None:
            port_id = f"port:{port}" if not str(port).startswith("port:") else str(port)
            if self.store.get_node(port_id) is None:
                self.store.upsert_node(Node(id=port_id, type="Port", ontology=ONTOLOGY_INFRA,
                                            props={"label": port_id, "number": int(port),
                                                   "protocol": ticket.get("protocol", "tcp"),
                                                   "riskLevel": "unknown"},
                                            provenance=Provenance(source=f"ticket:{case_id}",
                                                                  confidence=0.5)))
            self.store.upsert_edge(Edge(case_id, "port", port_id, ontology=ONTOLOGY_TICKET,
                                        provenance=Provenance(source=f"ticket:{case_id}")))
        self.store.upsert_edge(Edge(case_id, "reviewedBy", self.operator, ontology=ONTOLOGY_TICKET,
                                    provenance=Provenance(source="agent")))
        return case_id

    def _upsert_reference(self, value: Any, concept: str, ontology: str, case_id: str) -> str:
        if isinstance(value, dict):
            target_id = str(value.get("id") or value.get("label"))
            node = self.store.get_node(target_id)
            if node is None:
                node = Node(id=target_id, type=concept, ontology=ontology,
                            props={"label": target_id},
                            provenance=Provenance(source=f"ticket:{case_id}", confidence=0.6))
            for key, attribute in value.items():
                if key != "id":
                    node.props[key] = attribute
            self.store.upsert_node(node)
            return target_id
        target_id = str(value)
        if self.store.get_node(target_id) is None:
            self.store.upsert_node(Node(id=target_id, type=concept, ontology=ontology,
                                        props={"label": target_id},
                                        provenance=Provenance(source=f"ticket:{case_id}",
                                                              confidence=0.5)))
        return target_id

    # --- the loop --------------------------------------------------------
    def review(self, ticket: Dict[str, Any]) -> ReviewResult:
        """Run the review skill over one ticket."""
        case_id = self.ingest_ticket(ticket)
        ctx = SkillContext(situation=None, case_id=case_id, ticket=ticket, missing_paths=[])
        self.runtime.run(self.skill, ctx)
        situation: SituationABox = ctx["situation"]
        decision: Decision = ctx["decision"]
        gaps = OntologyGapDetector(self.tbox).from_graph(situation.graph, case_id)
        result = ReviewResult(
            case_id=case_id,
            decision=decision,
            situation=situation,
            trace=[entry.describe() for entry in ctx.trace],
            checks={"ownership": ctx.get("ownership"), "network": ctx.get("network"),
                    "security": ctx.get("security"), "vendor_risk": ctx.get("vendor_risk")},
            tbox_violations=tuple(self.tbox.validate_graph(situation.graph)),
            tbox_proposals=tuple(gaps),
            skill_version=self.skill.version,
        )
        self.open_cases[case_id] = result
        for proposal in gaps:
            self.proposals[proposal.id] = proposal
            record_proposal(self.store, "tbox", proposal.id, proposal.to_dict())
        return result

    def record_human_decision(self, case_id: str, human_decision: Any, reason: str = "",
                              approver: str = "") -> LearningOutcome:
        """
        Close a case and turn any mismatch into proposals.

        This is the reflection half of the KSTAR cycle: the case is archived, the
        delta is root-caused, and each asset that could explain the gap produces
        a proposal of its own.
        """
        result = self.open_cases.get(case_id)
        if result is None:
            raise KeyError(f"unknown or already closed case: {case_id}")
        human_action = Action.coerce(human_decision)
        delta = DecisionDelta(case_id=case_id, system_action=result.action,
                              human_action=human_action, reason=reason,
                              situation=result.situation, decision=result.decision)
        rule_proposals: Tuple[RuleChangeProposal, ...] = ()
        skill_proposals: Tuple[SkillChangeProposal, ...] = ()
        tbox_proposals: List[TBoxEvolutionProposal] = list(result.tbox_proposals)
        if delta.system_action != delta.human_action:
            analyzer = CounterfactualAnalyzer(self.archive,
                                              ignore_paths=("justification", "openedAt", "duration"))
            rule_proposals = tuple(analyzer.analyze(delta, self.rbox))
            for proposal in rule_proposals:
                self.gate.submit(proposal, self.rbox)
                self.proposals[proposal.id] = proposal
            skill_proposals = tuple(self._skill_proposals(delta))
            for proposal in skill_proposals:
                self.proposals[proposal.id] = proposal
                record_proposal(self.store, "skill", proposal.id, proposal.to_dict())
        self.archive.add(CaseRecord(case_id=case_id, situation=result.situation,
                                    system_action=result.action, human_action=human_action,
                                    human_reason=reason, rbox_version=self.rbox.version))
        del self.open_cases[case_id]
        if approver:
            self.observe_preference(approver, human_action, result)
        promotions = self._promote_preferences()
        for proposal in tbox_proposals:
            self.proposals[proposal.id] = proposal
        return LearningOutcome(delta=delta, rule_proposals=rule_proposals,
                               tbox_proposals=tuple(tbox_proposals),
                               skill_proposals=skill_proposals,
                               preference_promotions=tuple(promotions))

    def _skill_proposals(self, delta: DecisionDelta) -> List[SkillChangeProposal]:
        """Propose a procedure change when a whole class of request is unassessed."""
        proposals: List[SkillChangeProposal] = []
        organization = delta.situation.value("requester.organizationType")
        already_branches = any(node.id == "s5b_vendor_branch" for node in self.skill.steps())
        if organization == "ExternalVendor" and not already_branches:
            evidence = tuple(case.case_id for case in self.archive.cases
                             if case.situation.value("requester.organizationType") == "ExternalVendor")
            proposals.append(SkillChangeProposal(
                id=f"skill-proposal-{delta.case_id}",
                skill_id=self.skill.id,
                base_version=self.skill.version,
                operation="insert_after",
                anchor_step_id="s5_ownership_check",
                node=vendor_risk_branch(),
                rationale=("External vendor requests need a risk assessment before the "
                           "security policy check; the gap is in the procedure, not the rules."),
                evidence=(delta.case_id, *evidence),
            ))
        return proposals

    # --- governance ------------------------------------------------------
    def approve(self, proposal_id: str, approver: str, override_reason: str = "") -> str:
        """
        Apply an approved proposal and derive the next asset version.

        Returns:
            A human-readable summary of the version change.
        """
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"unknown proposal: {proposal_id}")
        if isinstance(proposal, RuleChangeProposal):
            previous = self.rbox.version
            self.rbox = self.gate.approve(proposal, self.rbox, approver, override_reason)
            return (f"RBox {RBOX_NAME} v{previous} -> v{self.rbox.version} "
                    f"({proposal.proposed_rule.describe()})")
        if isinstance(proposal, TBoxEvolutionProposal):
            previous = self.tbox.version
            self.tbox = proposal.apply(self.tbox)
            self.tbox.save(self.store, changelog=f"{proposal.id} approved by {approver}",
                           parent_version=previous)
            return f"TBox {TBOX_NAME} v{previous} -> v{self.tbox.version} ({proposal.observed_gap})"
        if isinstance(proposal, SkillChangeProposal):
            previous = self.skill.version
            self.skill = proposal.apply(self.skill)
            self.skill.save(self.store, changelog=f"{proposal.id} approved by {approver}",
                            parent_version=previous)
            return f"Skill {SKILL_ID} v{previous} -> v{self.skill.version} ({proposal.rationale})"
        raise TypeError(f"unsupported proposal type: {type(proposal).__name__}")

    def pending_proposals(self) -> List[str]:
        return [f"{key}: {getattr(value, 'status', 'proposed')}"
                for key, value in self.proposals.items()]

    # --- preferences -----------------------------------------------------
    def observe_preference(self, approver: str, human_action: Action, result: ReviewResult) -> None:
        """A reviewer who keeps escalating production changes is stating a preference."""
        environment = result.situation.value("destination.environment")
        if environment != "Production":
            return
        mode = "Conservative" if human_action.severity > Action.AUTO_APPROVE.severity else "Permissive"
        self.preferences.observe(PreferenceSignal(
            subject=approver, predicate="prefersApprovalMode", value=mode,
            scope="ProductionFirewallChange", source=f"case:{result.case_id}"))

    def _promote_preferences(self) -> List[str]:
        promoted, proposals = self.preferences.promote(self.store, self.tbox)
        for proposal in proposals:
            self.proposals[proposal.id] = proposal
            record_proposal(self.store, "tbox", proposal.id, proposal.to_dict())
        return [candidate.describe() for candidate in promoted]

    # --- skill operations ------------------------------------------------
    def _operations(self) -> Dict[str, Any]:
        return {
            "validate_input": self._op_validate_input,
            "resolve_entities": self._op_resolve_entities,
            "construct_situation": self._op_construct_situation,
            "retrieve_rbox": self._op_retrieve_rbox,
            "run_ownership_check": self._op_ownership_check,
            "run_network_check": self._op_network_check,
            "run_security_policy_check": self._op_security_check,
            "retrieve_missing_evidence": self._op_retrieve_missing_evidence,
            "evaluate_rules": self._op_evaluate_rules,
            "generate_recommendation": self._op_generate_recommendation,
            "generate_evidence_graph": self._op_generate_evidence_graph,
            "request_human_decision": self._op_request_human_decision,
            "learn_from_outcome": self._op_learn_from_outcome,
            "vendor_risk_assessment": self._op_vendor_risk_assessment,
        }

    def _op_validate_input(self, ctx: SkillContext, params: Dict[str, Any]) -> Dict[str, Any]:
        ticket = ctx["ticket"]
        required = ("destination", "port")
        missing = [field_name for field_name in required if not ticket.get(field_name)]
        return {"ok": not missing, "missing_fields": missing}

    def _op_resolve_entities(self, ctx: SkillContext, params: Dict[str, Any]) -> Dict[str, Any]:
        case_id = ctx["case_id"]
        unresolved = []
        for edge in self.store.out_edges(case_id):
            target = self.store.get_node(edge.dst)
            if target is None:
                unresolved.append(edge.dst)
            elif target.provenance.confidence < 0.7:
                unresolved.append(f"{target.id} (low confidence, asserted by the ticket)")
        return {"unresolved": unresolved}

    def _op_construct_situation(self, ctx: SkillContext, params: Dict[str, Any]) -> Dict[str, int]:
        builder = SituationBuilder(self.store, ContextSpec(hops=3))
        situation = builder.build(ctx["case_id"], case_id=ctx["case_id"],
                                  extra_roots=("task:FirewallChangeRequest",))
        ctx["situation"] = situation
        ctx.situation = situation
        return dict(situation.sources)

    def _op_retrieve_rbox(self, ctx: SkillContext, params: Dict[str, Any]) -> List[str]:
        return [rule.id for rule in self.rbox.applicable(ctx["situation"])]

    def _op_ownership_check(self, ctx: SkillContext, params: Dict[str, Any]) -> str:
        situation: SituationABox = ctx["situation"]
        authority = situation.value("requester.approvalAuthority")
        requester_team = situation.values("requester.memberOf")
        owner_team = situation.values("application.owner")
        if authority is None:
            self._note_missing(ctx, "requester.approvalAuthority")
            return "unknown: requester authority is not in the organizational ontology"
        if owner_team and requester_team and not set(requester_team) & set(owner_team):
            return f"requester is outside the owning team ({owner_team})"
        return f"requester authority {authority}, owning team {owner_team or 'unknown'}"

    def _op_network_check(self, ctx: SkillContext, params: Dict[str, Any]) -> str:
        situation: SituationABox = ctx["situation"]
        source_zone = situation.value("source.networkZone")
        destination_zone = situation.value("destination.networkZone")
        if source_zone is None:
            self._note_missing(ctx, "source.networkZone")
        if destination_zone is None:
            self._note_missing(ctx, "destination.networkZone")
        return f"{source_zone or '?'} -> {destination_zone or '?'}"

    def _op_security_check(self, ctx: SkillContext, params: Dict[str, Any]) -> str:
        situation: SituationABox = ctx["situation"]
        classification = situation.value("destination.dataClassification", "unknown")
        risk = situation.value("port.riskLevel", "unknown")
        if risk == "unknown":
            self._note_missing(ctx, "port.riskLevel")
        return f"data {classification}, port risk {risk}"

    def _op_retrieve_missing_evidence(self, ctx: SkillContext, params: Dict[str, Any]) -> str:
        """Loop body: try to fill one unresolved path from the store."""
        path = ctx.get("missing_path")
        situation: SituationABox = ctx["situation"]
        if not path:
            return "nothing to retrieve"
        head = path.split(".")[0]
        targets = [b.node_id for b in situation.resolve(head)]
        recovered = 0
        for node_id in targets:
            node = self.store.get_node(node_id)
            if node is None:
                continue
            situation.graph.add_node(node)
            for edge in self.store.out_edges(node_id):
                target = self.store.get_node(edge.dst)
                if target is not None:
                    situation.graph.add_node(target)
                    situation.graph.add_edge(edge)
                    recovered += 1
        return f"{path}: {recovered} additional facts pulled from Semantica"

    def _op_evaluate_rules(self, ctx: SkillContext, params: Dict[str, Any]) -> Decision:
        decision = self.rbox.evaluate(ctx["situation"])
        for path in decision.missing_paths:
            self._note_missing(ctx, path)
        return decision

    def _op_generate_recommendation(self, ctx: SkillContext, params: Dict[str, Any]) -> str:
        decision: Decision = ctx["decision"]
        return decision.action.value

    def _op_generate_evidence_graph(self, ctx: SkillContext, params: Dict[str, Any]) -> int:
        decision: Decision = ctx["decision"]
        self.store.load_graph(decision.evidence_graph)
        return len(decision.evidence_graph)

    def _op_request_human_decision(self, ctx: SkillContext, params: Dict[str, Any]) -> bool:
        decision: Decision = ctx["decision"]
        return decision.action != Action.AUTO_APPROVE

    def _op_learn_from_outcome(self, ctx: SkillContext, params: Dict[str, Any]) -> str:
        return "awaiting human decision"

    def _op_vendor_risk_assessment(self, ctx: SkillContext, params: Dict[str, Any]) -> str:
        situation: SituationABox = ctx["situation"]
        organization = situation.value("requester.organizationType", "unknown")
        duration = situation.value("duration", "unspecified")
        return f"vendor access ({organization}, duration {duration}) requires a named sponsor"

    def _note_missing(self, ctx: SkillContext, path: str) -> None:
        missing = ctx.get("missing_paths") or []
        if path not in missing:
            missing.append(path)
        ctx["missing_paths"] = missing
