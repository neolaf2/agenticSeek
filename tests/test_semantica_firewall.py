import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.semantica.abox import (ContextSpec, PreferenceLearner, PreferenceSignal,
                                    SituationBuilder)
from sources.semantica.conditions import All, Any_, Cmp, Not, parse_condition
from sources.semantica.evolution import CaseArchive, CaseRecord, GovernanceGate, candidate_paths
from sources.semantica.firewall import (FirewallReviewPipeline, RBOX_NAME, SKILL_ID, TBOX_NAME,
                                        build_rbox, build_skill, build_tbox, seed_ontologies)
from sources.semantica.model import AssetKind, Edge, Graph, Node
from sources.semantica.rbox import Action, RBox, Rule
from sources.semantica.skills import (Branch, Loop, SkillContext, SkillGraph, SkillRuntime,
                                      Sequence, Step)
from sources.semantica.store import HttpSemanticaStore, InMemorySemanticaStore
from sources.semantica.tbox import OntologyGapDetector

INTERNAL_TICKET = {"id": "FW-T1", "requester": "alice", "application": "PaymentApp",
                   "source": "10.23.1.0/24", "destination": "DB-PROD-07", "port": 5432,
                   "justification": "app needs its database", "duration": "permanent"}

VENDOR_TICKET = {"id": "FW-T2",
                 "requester": {"id": "vendor-carol", "role": "IntegrationEngineer",
                               "approvalAuthority": "Level2", "department": "VendorOps",
                               "organizationType": "ExternalVendor"},
                 "application": "PaymentApp", "source": "10.90.4.0/24",
                 "destination": "APP-PROD-11", "port": 443,
                 "justification": "vendor support tunnel", "duration": "90 days"}

SECOND_VENDOR_TICKET = {"id": "FW-T3",
                        "requester": {"id": "vendor-dan", "role": "SupportAnalyst",
                                      "approvalAuthority": "Level3",
                                      "department": "PartnerIntegration",
                                      "organizationType": "ExternalVendor"},
                        "application": "PaymentApp", "source": "10.90.4.0/24",
                        "destination": "APP-PROD-11", "port": 443,
                        "justification": "reconciliation", "duration": "30 days"}


class TestGraphResolution(unittest.TestCase):
    """Path resolution is the glue between the graph and the condition language."""

    def setUp(self):
        self.graph = Graph(name="g", root_id="ticket")
        self.graph.add_node(Node("ticket", "FirewallChangeRequest", {"duration": "90 days"}, "ticket"))
        self.graph.add_node(Node("srv", "Server", {"label": "DB-PROD-07",
                                                   "dataClassification": "Restricted"}, "infra"))
        self.graph.add_node(Node("zone", "NetworkZone", {"label": "PCI-Zone",
                                                         "trustLevel": "restricted"}, "infra"))
        self.graph.add_edge(Edge("ticket", "destination", "srv", ontology="ticket"))
        self.graph.add_edge(Edge("srv", "networkZone", "zone", ontology="infra"))

    def test_resolves_property_on_root(self):
        self.assertEqual([b.value for b in self.graph.resolve("duration")], ["90 days"])

    def test_resolves_relation_to_node_label(self):
        self.assertEqual([b.value for b in self.graph.resolve("destination.networkZone")], ["PCI-Zone"])

    def test_resolves_two_hop_property(self):
        bindings = self.graph.resolve("destination.networkZone.trustLevel")
        self.assertEqual([b.value for b in bindings], ["restricted"])

    def test_resolves_node_type(self):
        self.assertEqual([b.value for b in self.graph.resolve("destination.#type")], ["Server"])

    def test_missing_path_yields_no_bindings(self):
        self.assertEqual(self.graph.resolve("destination.nonexistent"), [])

    def test_binding_keeps_originating_node(self):
        binding = self.graph.resolve("destination.dataClassification")[0]
        self.assertEqual(binding.node_id, "srv")


class TestConditions(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(name="g", root_id="t")
        self.graph.add_node(Node("t", "T", {"zone": "PCI-Zone", "risk": 7}, "o"))

    def test_eq_and_ne(self):
        self.assertTrue(Cmp("zone", "eq", "PCI-Zone").evaluate(self.graph).matched)
        self.assertFalse(Cmp("zone", "ne", "PCI-Zone").evaluate(self.graph).matched)

    def test_ne_is_false_when_path_is_missing(self):
        """Absence is expressed by 'missing', never by a negative comparison."""
        self.assertFalse(Cmp("absent", "ne", "x").evaluate(self.graph).matched)
        self.assertTrue(Cmp("absent", "missing").evaluate(self.graph).matched)

    def test_numeric_operators(self):
        self.assertTrue(Cmp("risk", "gte", 7).evaluate(self.graph).matched)
        self.assertFalse(Cmp("risk", "lt", 3).evaluate(self.graph).matched)

    def test_boolean_composition(self):
        condition = All([Cmp("zone", "eq", "PCI-Zone"),
                         Any_([Cmp("risk", "gt", 10), Cmp("risk", "eq", 7)])])
        self.assertTrue(condition.evaluate(self.graph).matched)
        self.assertFalse(Not(condition).evaluate(self.graph).matched)

    def test_evidence_is_collected(self):
        result = All([Cmp("zone", "eq", "PCI-Zone"), Cmp("risk", "eq", 7)]).evaluate(self.graph)
        self.assertEqual({b.path for b in result.evidence}, {"zone", "risk"})

    def test_round_trip_serialisation(self):
        condition = All([Cmp("a", "in", ["x"]), Not(Cmp("b", "exists"))])
        rebuilt = parse_condition(condition.to_dict())
        self.assertEqual(rebuilt.to_dict(), condition.to_dict())


class TestStore(unittest.TestCase):
    def test_asset_versions_are_ordered_and_immutable(self):
        store = InMemorySemanticaStore()
        store.put_asset(AssetKind.RBOX, "p", "1.0", {"rules": []}, changelog="initial")
        store.put_asset(AssetKind.RBOX, "p", "1.1", {"rules": [1]}, changelog="amended")
        self.assertEqual([r.version for r in store.list_asset_versions(AssetKind.RBOX, "p")],
                         ["1.0", "1.1"])
        self.assertEqual(store.get_asset(AssetKind.RBOX, "p").version, "1.1")
        self.assertEqual(store.get_asset(AssetKind.RBOX, "p", "1.0").payload, {"rules": []})

    def test_payload_is_copied_on_write(self):
        store = InMemorySemanticaStore()
        payload = {"rules": []}
        store.put_asset(AssetKind.TBOX, "t", "1.0", payload)
        payload["rules"].append("mutated")
        self.assertEqual(store.get_asset(AssetKind.TBOX, "t").payload, {"rules": []})

    def test_ontology_namespaces(self):
        store = InMemorySemanticaStore()
        seed_ontologies(store)
        infra = store.graph("infrastructure")
        self.assertTrue(all(n.ontology == "infrastructure" for n in infra.walk()))
        self.assertIsNotNone(infra.get("DB-PROD-07"))
        self.assertIsNone(infra.get("alice"))

    def test_http_adapter_shapes_requests(self):
        class FakeResponse:
            content = b"{}"

            def raise_for_status(self):
                return None

            def json(self):
                return {}

        class FakeSession:
            def __init__(self):
                self.calls = []

            def request(self, method, url, **kwargs):
                self.calls.append((method, url, kwargs))
                return FakeResponse()

        session = FakeSession()
        store = HttpSemanticaStore("https://semantica.example/api/", session=session)
        store.upsert_node(Node("n1", "Server", {"label": "n1"}, "infra"))
        method, url, kwargs = session.calls[0]
        self.assertEqual((method, url), ("POST", "https://semantica.example/api/nodes"))
        self.assertEqual(kwargs["json"]["id"], "n1")


class TestTBox(unittest.TestCase):
    def setUp(self):
        self.tbox = build_tbox()

    def test_inherited_properties(self):
        self.assertIn("label", self.tbox.properties_of("Person"))
        self.assertTrue(self.tbox.is_a("Person", "Thing"))

    def test_validation_flags_undeclared_property(self):
        node = Node("x", "Person", {"organizationType": "ExternalVendor"}, "org")
        violations = self.tbox.validate_node(node)
        self.assertEqual(len(violations), 1)
        self.assertIn("organizationType", violations[0])

    def test_validation_flags_disallowed_value(self):
        node = Node("x", "Person", {"approvalAuthority": "Level9"}, "org")
        self.assertTrue(any("not in allowed set" in v for v in self.tbox.validate_node(node)))

    def test_gap_detector_proposes_property(self):
        graph = Graph(name="g")
        graph.add_node(Node("x", "Person", {"organizationType": "ExternalVendor"}, "org"))
        proposals = OntologyGapDetector(self.tbox).from_graph(graph, "FW-1")
        self.assertEqual(len(proposals), 1)
        self.assertIn("organizationType", proposals[0].observed_gap)

    def test_gap_detector_proposes_concept(self):
        graph = Graph(name="g")
        graph.add_node(Node("x", "TemporaryAccessException", {"validUntil": "2026-01-01"}, "org"))
        proposal = OntologyGapDetector(self.tbox).from_graph(graph, "FW-1")[0]
        self.assertEqual(proposal.proposed_concept.name, "TemporaryAccessException")
        self.assertIn("validUntil", proposal.proposed_concept.properties)

    def test_apply_bumps_version_without_mutating_the_base(self):
        proposal = OntologyGapDetector(self.tbox).from_missing_path("Person", "organizationType")
        evolved = proposal.apply(self.tbox)
        self.assertEqual(self.tbox.version, "1.0")
        self.assertEqual(evolved.version, "1.1")
        self.assertIn("organizationType", evolved.properties_of("Person"))
        self.assertEqual(proposal.status, "applied")


class TestSituationABox(unittest.TestCase):
    """A situation is assembled from several ontologies, not read from one."""

    def setUp(self):
        self.store = InMemorySemanticaStore()
        seed_ontologies(self.store)
        self.pipeline = FirewallReviewPipeline(store=self.store, seed=False)
        self.case_id = self.pipeline.ingest_ticket(INTERNAL_TICKET)
        self.situation = SituationBuilder(self.store, ContextSpec(hops=3)).build(
            self.case_id, case_id=self.case_id, extra_roots=("task:FirewallChangeRequest",))

    def test_pulls_from_every_ontology(self):
        self.assertGreaterEqual(set(self.situation.sources),
                                {"ticket", "organizational", "infrastructure", "task"})

    def test_facts_resolve_across_ontology_boundaries(self):
        self.assertEqual(self.situation.value("destination.networkZone"), "PCI-Zone")
        self.assertEqual(self.situation.value("requester.approvalAuthority"), "Level2")
        self.assertEqual(self.situation.value("application.businessCriticality"), "Tier1")

    def test_provenance_records_the_source_ontology(self):
        self.assertIn("infrastructure", self.situation.explain("DB-PROD-07"))

    def test_hop_limit_is_respected(self):
        narrow = SituationBuilder(self.store, ContextSpec(hops=1)).build(self.case_id)
        self.assertIsNone(narrow.value("destination.networkZone.trustLevel"))

    def test_situation_is_self_contained(self):
        for edge in self.situation.graph.edges.values():
            self.assertIn(edge.src, self.situation.graph.nodes)
            self.assertIn(edge.dst, self.situation.graph.nodes)


class TestRBox(unittest.TestCase):
    def setUp(self):
        self.pipeline = FirewallReviewPipeline()

    def test_internal_pci_request_needs_security_review(self):
        result = self.pipeline.review(INTERNAL_TICKET)
        self.assertEqual(result.action, Action.SECURITY_REVIEW)
        self.assertIn("R102", [hit.rule_id for hit in result.decision.hits])

    def test_most_restrictive_action_wins(self):
        rbox = RBox("t", "1.0", [
            Rule("A", All([Cmp("x", "eq", 1)]), Action.AUTO_APPROVE),
            Rule("B", All([Cmp("x", "eq", 1)]), Action.REJECT),
            Rule("C", All([Cmp("x", "eq", 1)]), Action.MANUAL_REVIEW),
        ])
        graph = Graph(name="g", root_id="n")
        graph.add_node(Node("n", "T", {"x": 1}, "o"))
        self.assertEqual(rbox.evaluate(graph).action, Action.REJECT)

    def test_default_action_when_nothing_fires(self):
        result = self.pipeline.review({"id": "FW-T9", "requester": "alice",
                                       "application": "ReportingApp", "source": "10.23.1.0/24",
                                       "destination": "RPT-STG-02", "port": 443})
        self.assertEqual(result.action, Action.AUTO_APPROVE)

    def test_evidence_graph_links_decision_rule_and_fact(self):
        result = self.pipeline.review(INTERNAL_TICKET)
        graph = result.decision.evidence_graph
        rules = [n for n in graph.walk() if n.type == "Rule"]
        facts = [n for n in graph.walk() if n.type == "Fact"]
        self.assertTrue(rules and facts)
        self.assertTrue(any(e.rel == "supportedBy" for e in graph.edges.values()))
        self.assertTrue(any(e.rel == "assertedOn" for e in graph.edges.values()))

    def test_amendment_deprecates_the_previous_rule_version(self):
        rbox = build_rbox()
        amended = Rule("R102", All([Cmp("x", "eq", 1)]), Action.SECURITY_REVIEW, version="1.1")
        evolved = rbox.with_rule(amended)
        self.assertEqual(evolved.version, "1.1")
        self.assertEqual(rbox.version, "1.0")
        self.assertEqual(evolved.get("R102").version, "1.1")
        self.assertEqual(evolved.get("R102@1.0").status, "deprecated")

    def test_round_trip_serialisation(self):
        rbox = build_rbox()
        self.assertEqual(RBox.from_dict(rbox.to_dict()).to_dict(), rbox.to_dict())


class TestSkillGraph(unittest.TestCase):
    def setUp(self):
        self.runtime = SkillRuntime({
            "note": lambda ctx, params: ctx.values.setdefault("notes", []).append(params["text"]),
            "count": lambda ctx, params: ctx.values.__setitem__("seen", ctx.get("seen", 0) + 1),
        })

    def test_sequence_condition_and_loop(self):
        skill = SkillGraph("s", "s", Sequence("root", [
            Step("a", "note", {"text": "first"}),
            Branch("b", Cmp("flag", "eq", True), then=Step("c", "note", {"text": "branch"})),
            Loop("d", over="items", var="item", body=Step("e", "count")),
        ]))
        ctx = SkillContext(flag=True, items=["x", "y", "z"])
        run = self.runtime.run(skill, ctx)
        self.assertEqual(ctx["notes"], ["first", "branch"])
        self.assertEqual(ctx["seen"], 3)
        self.assertTrue(any(entry.step_id == "d" for entry in run.trace))

    def test_branch_reads_the_situation_when_context_is_silent(self):
        graph = Graph(name="g", root_id="n")
        graph.add_node(Node("n", "T", {"organizationType": "ExternalVendor"}, "o"))
        skill = SkillGraph("s", "s", Sequence("root", [
            Branch("b", Cmp("organizationType", "eq", "ExternalVendor"),
                   then=Step("c", "note", {"text": "vendor"})),
        ]))
        ctx = SkillContext(situation=graph)
        self.runtime.run(skill, ctx)
        self.assertEqual(ctx["notes"], ["vendor"])

    def test_insert_after_bumps_version_without_mutating_the_base(self):
        skill = build_skill()
        evolved = skill.insert_after("s5_ownership_check", Step("new", "note"))
        self.assertEqual(skill.version, "1.0")
        self.assertEqual(evolved.version, "1.1")
        self.assertIsNone(skill.find("new"))
        ids = [node.id for node in evolved.steps()]
        self.assertEqual(ids[ids.index("s5_ownership_check") + 1], "new")

    def test_unknown_step_raises(self):
        with self.assertRaises(KeyError):
            build_skill().insert_after("does-not-exist", Step("new", "note"))

    def test_round_trip_serialisation(self):
        skill = build_skill()
        self.assertEqual(SkillGraph.from_dict(skill.to_dict()).to_dict(), skill.to_dict())


class TestPreferenceLearning(unittest.TestCase):
    """Episodic memory becomes a Personal ABox fact, and may need a TBox extension."""

    def test_confidence_grows_with_repetition(self):
        learner = PreferenceLearner()
        for index in range(3):
            learner.observe(PreferenceSignal("richard", "prefersApprovalMode", "Conservative",
                                             "ProductionFirewallChange", f"session_{index}"))
        candidate = learner.candidates()[0]
        self.assertAlmostEqual(candidate.confidence, 0.875, places=3)
        self.assertEqual(len(candidate.sources), 3)

    def test_contradiction_lowers_confidence(self):
        learner = PreferenceLearner()
        for index in range(3):
            learner.observe(PreferenceSignal("richard", "prefersApprovalMode", "Conservative",
                                             "s", f"session_{index}"))
        learner.observe(PreferenceSignal("richard", "prefersApprovalMode", "Permissive",
                                         "s", "session_9"))
        candidate = learner.candidates()[0]
        self.assertEqual(candidate.contradictions, 1)
        self.assertLess(candidate.confidence, 0.875)

    def test_promotion_writes_the_fact_and_proposes_the_concept(self):
        store = InMemorySemanticaStore()
        seed_ontologies(store)
        learner = PreferenceLearner(threshold=0.8)
        for index in range(3):
            learner.observe(PreferenceSignal("richard", "prefersApprovalMode", "Conservative",
                                             "ProductionFirewallChange", f"session_{index}"))
        promoted, proposals = learner.promote(store, build_tbox())
        self.assertEqual(len(promoted), 1)
        self.assertEqual(store.get_node("richard").props["prefersApprovalMode"], "Conservative")
        self.assertEqual(len(proposals), 1)
        self.assertIn("prefersApprovalMode", proposals[0].observed_gap)

    def test_low_confidence_is_not_promoted(self):
        store = InMemorySemanticaStore()
        learner = PreferenceLearner(threshold=0.8)
        learner.observe(PreferenceSignal("richard", "prefersApprovalMode", "Conservative"))
        promoted, _ = learner.promote(store)
        self.assertEqual(promoted, [])


class TestCounterfactualLearning(unittest.TestCase):
    """The rule-learning loop: delta -> hypothesis -> back-test -> approval."""

    def setUp(self):
        self.pipeline = FirewallReviewPipeline()

    def _agreeing_history(self):
        for ticket, human in ((INTERNAL_TICKET, "SECURITY_REVIEW"),
                              ({"id": "FW-H2", "requester": "bob", "application": "PaymentApp",
                                "source": "10.90.4.0/24", "destination": "APP-PROD-11",
                                "port": 443}, "MANUAL_REVIEW"),
                              ({"id": "FW-H3", "requester": "alice", "application": "ReportingApp",
                                "source": "10.23.1.0/24", "destination": "RPT-STG-02",
                                "port": 443}, "AUTO_APPROVE")):
            result = self.pipeline.review(ticket)
            self.pipeline.record_human_decision(result.case_id, human, "baseline")

    def test_candidate_paths_include_unused_facts(self):
        result = self.pipeline.review(VENDOR_TICKET)
        paths = candidate_paths(result.situation)
        self.assertIn("requester.organizationType", paths)
        self.assertIn("destination.networkZone", paths)

    def test_vendor_case_is_not_explained_by_the_initial_rules(self):
        result = self.pipeline.review(VENDOR_TICKET)
        self.assertEqual(result.action, Action.AUTO_APPROVE)

    def test_two_diverse_cases_generalise_to_the_shared_attribute(self):
        self._agreeing_history()
        for ticket in (VENDOR_TICKET, SECOND_VENDOR_TICKET):
            result = self.pipeline.review(ticket)
            outcome = self.pipeline.record_human_decision(
                result.case_id, "SECURITY_REVIEW",
                "External vendor access requires review regardless of network zone.")
        best = outcome.rule_proposals[0]
        self.assertIn("requester.organizationType", best.proposed_rule.when.describe())
        self.assertIn("ExternalVendor", best.proposed_rule.when.describe())
        self.assertEqual(best.base_rule_id, "R102")
        self.assertEqual(len(best.backtest.fixed), 2)
        self.assertEqual(best.backtest.regressed, ())

    def test_amendment_keeps_the_original_condition(self):
        self._agreeing_history()
        result = self.pipeline.review(VENDOR_TICKET)
        outcome = self.pipeline.record_human_decision(result.case_id, "SECURITY_REVIEW", "vendor")
        described = outcome.rule_proposals[0].proposed_rule.when.describe()
        self.assertIn("destination.networkZone = 'PCI-Zone'", described)
        self.assertIn("OR", described)

    def test_nothing_is_proposed_when_the_human_agrees(self):
        result = self.pipeline.review(INTERNAL_TICKET)
        outcome = self.pipeline.record_human_decision(result.case_id, "SECURITY_REVIEW", "agreed")
        self.assertEqual(outcome.rule_proposals, ())
        self.assertFalse(outcome.has_proposals)

    def test_a_regressing_proposal_is_never_returned(self):
        archive = CaseArchive()
        pipeline = FirewallReviewPipeline(archive=archive)
        result = pipeline.review(INTERNAL_TICKET)
        pipeline.record_human_decision(result.case_id, "SECURITY_REVIEW", "baseline")
        delta_case = pipeline.review(dict(INTERNAL_TICKET, id="FW-T1b"))
        outcome = pipeline.record_human_decision(delta_case.case_id, "AUTO_APPROVE",
                                                 "this one is fine")
        for proposal in outcome.rule_proposals:
            self.assertEqual(proposal.backtest.regressed, ())

    def test_approval_is_required_and_produces_a_new_version(self):
        self._agreeing_history()
        result = self.pipeline.review(VENDOR_TICKET)
        outcome = self.pipeline.record_human_decision(result.case_id, "SECURITY_REVIEW", "vendor")
        proposal = outcome.rule_proposals[0]
        with self.assertRaises(PermissionError):
            self.pipeline.gate.approve(proposal, self.pipeline.rbox, approver="")
        summary = self.pipeline.approve(proposal.id, approver="richard")
        self.assertIn("v1.0 -> v1.1", summary)
        self.assertEqual(self.pipeline.rbox.version, "1.1")

    def test_unsafe_proposal_requires_an_override(self):
        archive = CaseArchive()
        gate = GovernanceGate(archive)
        rbox = build_rbox()
        pipeline = FirewallReviewPipeline(archive=archive)
        result = pipeline.review(INTERNAL_TICKET)
        archive.add(CaseRecord("FW-past", result.situation, Action.SECURITY_REVIEW,
                               Action.SECURITY_REVIEW))
        breaking = Rule("R900", All([Cmp("destination.networkZone", "eq", "PCI-Zone")]),
                        Action.REJECT, version="1.0", priority=1)
        from sources.semantica.evolution import RuleChangeProposal
        proposal = RuleChangeProposal(id="p1", kind="new", proposed_rule=breaking)
        gate.submit(proposal, rbox)
        self.assertFalse(proposal.backtest.is_safe)
        with self.assertRaises(PermissionError):
            gate.approve(proposal, rbox, approver="richard")
        evolved = gate.approve(proposal, rbox, approver="richard",
                               override_reason="accepted risk, ticket SEC-42")
        self.assertEqual(evolved.version, "1.1")

    def test_an_added_permissive_rule_cannot_loosen_an_outcome(self):
        """Most-restrictive-wins means a learned rule can only ever tighten."""
        pipeline = FirewallReviewPipeline()
        before = pipeline.review(INTERNAL_TICKET)
        permissive = Rule("R901", All([Cmp("destination.networkZone", "eq", "PCI-Zone")]),
                          Action.AUTO_APPROVE, version="1.0", priority=1)
        pipeline.rbox = pipeline.rbox.with_rule(permissive)
        after = pipeline.review(dict(INTERNAL_TICKET, id="FW-T1c"))
        self.assertEqual(before.action, Action.SECURITY_REVIEW)
        self.assertEqual(after.action, Action.SECURITY_REVIEW)

    def test_evolved_rbox_changes_the_outcome_of_the_same_request(self):
        self._agreeing_history()
        for ticket in (VENDOR_TICKET, SECOND_VENDOR_TICKET):
            result = self.pipeline.review(ticket)
            outcome = self.pipeline.record_human_decision(result.case_id, "SECURITY_REVIEW",
                                                          "external vendor")
        self.pipeline.approve(outcome.rule_proposals[0].id, approver="richard")
        replay = self.pipeline.review(dict(VENDOR_TICKET, id="FW-T4"))
        self.assertEqual(replay.action, Action.SECURITY_REVIEW)
        self.assertIn("R102", [hit.rule_id for hit in replay.decision.hits])


class TestSkillEvolution(unittest.TestCase):
    def setUp(self):
        self.pipeline = FirewallReviewPipeline()

    def test_vendor_case_proposes_a_procedure_change(self):
        result = self.pipeline.review(VENDOR_TICKET)
        outcome = self.pipeline.record_human_decision(result.case_id, "SECURITY_REVIEW", "vendor")
        self.assertEqual(len(outcome.skill_proposals), 1)
        proposal = outcome.skill_proposals[0]
        self.assertEqual(proposal.anchor_step_id, "s5_ownership_check")

    def test_approved_patch_runs_the_new_step(self):
        result = self.pipeline.review(VENDOR_TICKET)
        outcome = self.pipeline.record_human_decision(result.case_id, "SECURITY_REVIEW", "vendor")
        summary = self.pipeline.approve(outcome.skill_proposals[0].id, approver="richard")
        self.assertIn("v1.0 -> v1.1", summary)
        replay = self.pipeline.review(dict(VENDOR_TICKET, id="FW-T5"))
        self.assertIn("vendor access", replay.checks["vendor_risk"])
        self.assertTrue(any("s5c_vendor_risk_assessment" in entry for entry in replay.trace))

    def test_patch_is_not_proposed_twice(self):
        result = self.pipeline.review(VENDOR_TICKET)
        outcome = self.pipeline.record_human_decision(result.case_id, "SECURITY_REVIEW", "vendor")
        self.pipeline.approve(outcome.skill_proposals[0].id, approver="richard")
        again = self.pipeline.review(dict(VENDOR_TICKET, id="FW-T6"))
        outcome = self.pipeline.record_human_decision(again.case_id, "REJECT", "no sponsor")
        self.assertEqual(outcome.skill_proposals, ())

    def test_non_vendor_case_gets_no_skill_proposal(self):
        result = self.pipeline.review(INTERNAL_TICKET)
        outcome = self.pipeline.record_human_decision(result.case_id, "REJECT", "too broad")
        self.assertEqual(outcome.skill_proposals, ())


class TestPipelineReporting(unittest.TestCase):
    def setUp(self):
        self.pipeline = FirewallReviewPipeline()

    def test_review_records_assets_and_case(self):
        result = self.pipeline.review(INTERNAL_TICKET)
        self.assertIn(result.case_id, self.pipeline.open_cases)
        for kind, name in ((AssetKind.TBOX, TBOX_NAME), (AssetKind.RBOX, RBOX_NAME),
                           (AssetKind.SKILL, SKILL_ID)):
            self.assertIsNotNone(self.pipeline.store.get_asset(kind, name))

    def test_report_contains_evidence_and_trace(self):
        report = self.pipeline.review(INTERNAL_TICKET).report()
        self.assertIn("Recommendation", report)
        self.assertIn("R102", report)
        self.assertIn("Situation ABox", report)
        self.assertIn("s9_evaluate_rules", report)

    def test_unknown_entities_are_flagged_not_invented(self):
        result = self.pipeline.review({"id": "FW-T7", "requester": "eve",
                                       "destination": "UNKNOWN-HOST", "port": 9999,
                                       "source": "192.0.2.0/24"})
        self.assertTrue(result.checks["ownership"].startswith("unknown"))
        self.assertIn("?", result.checks["network"])

    def test_closing_an_unknown_case_raises(self):
        with self.assertRaises(KeyError):
            self.pipeline.record_human_decision("nope", "REJECT")

    def test_tbox_violation_is_reported_for_undeclared_vendor_attribute(self):
        result = self.pipeline.review(VENDOR_TICKET)
        self.assertTrue(any("organizationType" in violation for violation in result.tbox_violations))
        self.assertTrue(result.tbox_proposals)


if __name__ == "__main__":
    unittest.main()
