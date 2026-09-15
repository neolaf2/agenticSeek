#!/usr/bin/env python3
"""
End-to-end demonstration of the firewall CR review agent.

Runs the full loop without an LLM and without a network: review a ticket, take
the human decision, root-cause the mismatch, back-test the proposal, approve it,
and show the next version of each cognitive asset behaving differently.

    python3 examples/firewall_cr_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sources.semantica.firewall import FirewallReviewPipeline  # noqa: E402
from sources.semantica.model import AssetKind  # noqa: E402

HISTORY = [
    ({"id": "FW-10101", "requester": "alice", "application": "PaymentApp",
      "source": "10.23.1.0/24", "destination": "DB-PROD-07", "port": 5432,
      "justification": "PaymentApp needs its database", "duration": "permanent"},
     "SECURITY_REVIEW"),
    ({"id": "FW-10102", "requester": "bob", "application": "PaymentApp",
      "source": "10.90.4.0/24", "destination": "APP-PROD-11", "port": 443,
      "justification": "internal service call", "duration": "permanent"},
     "MANUAL_REVIEW"),
    ({"id": "FW-10103", "requester": "alice", "application": "ReportingApp",
      "source": "10.23.1.0/24", "destination": "RPT-STG-02", "port": 443,
      "justification": "reporting UI", "duration": "permanent"},
     "AUTO_APPROVE"),
]

VENDOR_CASES = [
    {"id": "FW-10293",
     "requester": {"id": "vendor-carol", "label": "Carol (Acme Ltd)", "role": "IntegrationEngineer",
                   "approvalAuthority": "Level2", "department": "VendorOps",
                   "organizationType": "ExternalVendor"},
     "application": "PaymentApp", "source": "10.90.4.0/24", "destination": "APP-PROD-11",
     "port": 443, "justification": "vendor support tunnel", "duration": "90 days"},
    {"id": "FW-10321",
     "requester": {"id": "vendor-dan", "label": "Dan (Globex)", "role": "SupportAnalyst",
                   "approvalAuthority": "Level3", "department": "PartnerIntegration",
                   "organizationType": "ExternalVendor"},
     "application": "PaymentApp", "source": "10.90.4.0/24", "destination": "APP-PROD-11",
     "port": 443, "justification": "quarterly reconciliation access", "duration": "30 days"},
]


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main() -> None:
    pipeline = FirewallReviewPipeline(operator="richard")

    banner("1. History: cases where the agent and the humans agreed")
    for ticket, human in HISTORY:
        result = pipeline.review(ticket)
        print(f"{result.case_id}: system={result.action.value:<16} human={human}")
        pipeline.record_human_decision(result.case_id, human, "baseline case", approver="richard")

    banner("2. A vendor case the rules cannot explain")
    outcome = None
    for ticket in VENDOR_CASES:
        result = pipeline.review(ticket)
        print(f"\n{result.case_id}: recommendation {result.action.value}")
        print(result.decision.explain())
        outcome = pipeline.record_human_decision(
            result.case_id, "SECURITY_REVIEW",
            "External vendor access requires review regardless of network zone.",
            approver="richard")
        print("\n" + outcome.report())

    banner("3. Governance: back-test, then approve")
    rule_proposal = outcome.rule_proposals[0]
    print(rule_proposal.describe())
    print("\n" + pipeline.approve(rule_proposal.id, approver="richard"))
    for proposal in outcome.skill_proposals:
        print(pipeline.approve(proposal.id, approver="richard"))
    for proposal in outcome.tbox_proposals:
        print(pipeline.approve(proposal.id, approver="richard"))

    banner("4. The same shape of request, after the assets evolved")
    replay = dict(VENDOR_CASES[0], id="FW-10402")
    result = pipeline.review(replay)
    print(f"{result.case_id}: recommendation {result.action.value}")
    print(result.decision.explain())
    print("\nvendor risk step:", result.checks.get("vendor_risk"))
    print("\nskill now reads:")
    print(pipeline.skill.outline())

    banner("5. Learned preference and asset versions in Semantica")
    for candidate in pipeline.preferences.candidates():
        print("preference:", candidate.describe())
    for kind, name in ((AssetKind.TBOX, "firewall-world-model"),
                       (AssetKind.RBOX, "firewall-policy"),
                       (AssetKind.SKILL, "EvaluateFirewallRequestSkill")):
        versions = pipeline.store.list_asset_versions(kind, name)
        trail = " -> ".join(f"v{record.version}" for record in versions)
        print(f"{kind.value:<6} {name}: {trail}")
        for record in versions:
            if record.changelog:
                print(f"         v{record.version}: {record.changelog}")


if __name__ == "__main__":
    main()
