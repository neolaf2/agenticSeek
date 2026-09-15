import asyncio
import importlib.util
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# The agent base class pulls in the project's heavy runtime (torch, transformers).
# The reasoning it wraps is covered without them in test_semantica_firewall.py.
HAS_RUNTIME = importlib.util.find_spec("torch") is not None

if HAS_RUNTIME:
    from sources.agents.firewall_cr_agent import FirewallCRAgent

PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'base',
                           'firewall_cr_agent.txt')

TICKET = {"id": "FW-A1", "requester": "alice", "application": "PaymentApp",
          "source": "10.23.1.0/24", "destination": "DB-PROD-07", "port": 5432}


class FakeProvider:
    """Stands in for an LLM provider; the agent must not need it to decide."""

    def __init__(self, answer: str = "") -> None:
        self.answer = answer
        self.calls = 0

    def get_model_name(self) -> str:
        return "fake-model"

    def respond(self, memory, verbose=False) -> str:
        self.calls += 1
        return self.answer


@unittest.skipUnless(HAS_RUNTIME, "agent runtime dependencies are not installed")
class TestFirewallCRAgent(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.agent = FirewallCRAgent("Firewall", PROMPT_PATH, self.provider)

    def ask(self, prompt: str) -> str:
        answer, _ = asyncio.run(self.agent.process(prompt, None))
        return answer

    def test_reviews_a_json_ticket_without_calling_the_llm(self):
        answer = self.ask(json.dumps(TICKET))
        self.assertIn("SECURITY_REVIEW", answer)
        self.assertIn("R102", answer)
        self.assertEqual(self.provider.calls, 0)

    def test_fenced_json_is_accepted(self):
        answer = self.ask(f"please review:\n```json\n{json.dumps(TICKET)}\n```")
        self.assertIn("Firewall CR review", answer)

    def test_decision_closes_the_case_and_reports_deltas(self):
        self.ask(json.dumps(TICKET))
        answer = self.ask(json.dumps({"case_id": "FW-A1", "human_decision": "REJECT",
                                      "reason": "the path is too broad"}))
        self.assertIn("Learning from FW-A1", answer)
        self.assertEqual(len(self.agent.pipeline.archive), 1)

    def test_status_lists_asset_versions(self):
        answer = self.ask("status")
        self.assertIn("firewall-policy v1.0", answer)
        self.assertIn("EvaluateFirewallRequestSkill", answer)

    def test_approve_requires_a_known_proposal(self):
        answer = self.ask("approve not-a-proposal")
        self.assertIn("Cannot approve", answer)
        self.assertFalse(self.agent.success)

    def test_free_text_falls_back_to_llm_extraction(self):
        self.provider.answer = json.dumps(dict(TICKET, id="FW-A2"))
        answer = self.ask("Alice needs PaymentApp to reach DB-PROD-07 on 5432.")
        self.assertEqual(self.provider.calls, 1)
        self.assertIn("FW-A2", answer)

    def test_unparseable_request_explains_the_interface(self):
        self.provider.answer = "I could not find a ticket."
        answer = self.ask("hello there")
        self.assertIn("I review firewall change requests", answer)


if __name__ == "__main__":
    unittest.main()
