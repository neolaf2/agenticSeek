import json
import re
from typing import Any, Dict, Optional, Tuple

from sources.agents.agent import Agent
from sources.memory import Memory
from sources.semantica.firewall import (FirewallReviewPipeline, RBOX_NAME, SKILL_ID, TBOX_NAME)
from sources.semantica.model import AssetKind
from sources.semantica.store import SemanticaStore, open_store
from sources.utility import animate_thinking, pretty_print

EXTRACTION_INSTRUCTION = """Extract the firewall change request from the message below as JSON.
Use exactly these keys, omitting any you cannot find:
{"id": str, "requester": str, "application": str, "source": str, "destination": str,
 "port": int, "protocol": str, "justification": str, "duration": str}
Answer with the JSON object only.

Message:
"""


class FirewallCRAgent(Agent):
    """
    Reviews firewall change requests against ontologies held in Semantica.

    The agent is a thin conversational shell over
    :class:`~sources.semantica.firewall.FirewallReviewPipeline`: the reasoning is
    the pipeline's, so every recommendation is reproducible without an LLM. The
    LLM is used only to turn a free-text request into a structured ticket.

    It understands four kinds of message:
        - a ticket (JSON, or free text the LLM can turn into a ticket)
        - a decision, ``{"case_id": ..., "human_decision": ..., "reason": ...}``
        - ``approve <proposal-id>`` / ``reject <proposal-id> <reason>``
        - ``status``
    """

    def __init__(self, name, prompt_path, provider, verbose=False,
                 store: Optional[SemanticaStore] = None, operator: str = "richard") -> None:
        super().__init__(name, prompt_path, provider, verbose, None)
        self.role = "security"
        self.type = "firewall_cr_agent"
        self.operator = operator
        self.pipeline = FirewallReviewPipeline(store=store or open_store(), operator=operator)
        self.tools = {}
        self.memory = Memory(self.load_prompt(prompt_path),
                             recover_last_session=False,
                             memory_compression=False,
                             model_provider=provider.get_model_name() if provider else "none")

    async def process(self, prompt: str, speech_module) -> Tuple[str, str]:
        """
        Route one message and answer with the resulting report.

        Returns:
            The answer and the reasoning text (empty when no LLM was involved).
        """
        self.memory.push('user', prompt)
        reasoning = ""
        decision = self._parse_decision(prompt)
        if decision is not None:
            answer = self._close_case(decision)
        elif self._is_governance(prompt):
            answer = self._govern(prompt)
        elif self._is_status(prompt):
            answer = self.status_report()
        else:
            ticket = self._parse_ticket(prompt)
            if ticket is None:
                animate_thinking("Extracting the change request...", color="status")
                ticket, reasoning = await self._extract_ticket(prompt)
            answer = self._review(ticket) if ticket else self._usage()
        self.last_answer = answer
        self.last_reasoning = reasoning
        self.memory.push('assistant', answer)
        self.status_message = "Ready"
        return answer, reasoning

    # --- routing ---------------------------------------------------------
    def _review(self, ticket: Dict[str, Any]) -> str:
        self.status_message = "Reviewing the change request..."
        result = self.pipeline.review(ticket)
        self.success = True
        return result.report()

    def _close_case(self, decision: Dict[str, Any]) -> str:
        self.status_message = "Learning from the decision..."
        try:
            outcome = self.pipeline.record_human_decision(
                case_id=str(decision["case_id"]),
                human_decision=decision.get("human_decision") or decision.get("decision"),
                reason=decision.get("reason", ""),
                approver=decision.get("approver", self.operator))
        except KeyError as error:
            self.success = False
            return f"No open case matches that decision: {error}"
        return outcome.report()

    def _govern(self, prompt: str) -> str:
        match = re.search(r"\b(approve|reject)\s+([\w:@.\-]+)\s*(.*)", prompt.strip(), re.IGNORECASE)
        if match is None:
            return self._usage()
        verb, proposal_id, remainder = match.group(1).lower(), match.group(2), match.group(3)
        if verb == "reject":
            proposal = self.pipeline.proposals.get(proposal_id)
            if proposal is None:
                return f"Unknown proposal: {proposal_id}"
            setattr(proposal, "status", "rejected")
            return f"{proposal_id} rejected: {remainder or 'no reason given'}"
        try:
            summary = self.pipeline.approve(proposal_id, approver=self.operator,
                                            override_reason=remainder.strip())
        except (KeyError, PermissionError) as error:
            self.success = False
            return f"Cannot approve {proposal_id}: {error}"
        return f"Approved by {self.operator}.\n\n{summary}"

    def status_report(self) -> str:
        """Current asset versions, open cases and pending proposals."""
        lines = ["# Semantica cognitive assets", ""]
        for kind, name in ((AssetKind.TBOX, TBOX_NAME), (AssetKind.RBOX, RBOX_NAME),
                           (AssetKind.SKILL, SKILL_ID)):
            versions = self.pipeline.store.list_asset_versions(kind, name)
            trail = " -> ".join(f"v{record.version}" for record in versions) or "v?"
            lines.append(f"- {kind.value}: {name} {trail}")
        lines += ["", f"open cases: {sorted(self.pipeline.open_cases) or 'none'}",
                  f"closed cases: {len(self.pipeline.archive)}", "", "## Pending proposals"]
        pending = self.pipeline.pending_proposals()
        if pending:
            lines.extend(f"- {entry}" for entry in pending)
        else:
            lines.append("- none")
        for candidate in self.pipeline.preferences.candidates():
            lines.append(f"- learned preference: {candidate.describe()}")
        return "\n".join(lines)

    # --- parsing ---------------------------------------------------------
    def _parse_ticket(self, prompt: str) -> Optional[Dict[str, Any]]:
        payload = self._parse_json(prompt)
        if payload is None:
            return None
        if any(key in payload for key in ("destination", "source", "port")):
            return payload
        return None

    def _parse_decision(self, prompt: str) -> Optional[Dict[str, Any]]:
        payload = self._parse_json(prompt)
        if payload is None:
            return None
        if "case_id" in payload and ("human_decision" in payload or "decision" in payload):
            return payload
        return None

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidates = [fenced.group(1)] if fenced else []
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            candidates.append(brace.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _is_governance(prompt: str) -> bool:
        return bool(re.match(r"^\s*(approve|reject)\b", prompt, re.IGNORECASE))

    @staticmethod
    def _is_status(prompt: str) -> bool:
        return bool(re.match(r"^\s*(status|assets|versions|pending)\b", prompt, re.IGNORECASE))

    async def _extract_ticket(self, prompt: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Fall back to the LLM to structure a free-text request."""
        if self.llm is None:
            return None, ""
        self.memory.push('user', EXTRACTION_INSTRUCTION + prompt)
        try:
            answer, reasoning = await self.llm_request()
        except Exception as error:
            if self.verbose:
                pretty_print(f"Ticket extraction failed: {error}", color="failure")
            return None, ""
        return self._parse_ticket(answer), reasoning

    def _usage(self) -> str:
        self.success = False
        return ("I review firewall change requests against the ontologies in Semantica.\n\n"
                "Send me one of:\n"
                '- a ticket: `{"id": "FW-10293", "requester": "alice", "application": "PaymentApp", '
                '"source": "10.23.1.0/24", "destination": "DB-PROD-07", "port": 5432}`\n'
                '- a decision: `{"case_id": "FW-10293", "human_decision": "SECURITY_REVIEW", '
                '"reason": "external vendor"}`\n'
                "- `approve <proposal-id>` or `reject <proposal-id> <reason>`\n"
                "- `status`")


if __name__ == "__main__":
    pass
