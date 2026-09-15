# Firewall CR Review and Approval Agent

An agent that reviews firewall change requests against ontologies held in
**Semantica**, recommends an action with an evidence graph behind it, and turns
every human decision that disagrees with it into a *proposal* to evolve one of
its own cognitive assets.

Nothing in the decision path is generated text. The LLM is used at one point
only: turning a free-text request into a structured ticket. Everything after
that is replayable, which is what makes the learning loop auditable.

```
sources/semantica/
    model.py        property graph: Node, Edge, Graph, Provenance, Version
    store.py        SemanticaStore: embedded backend + HTTP adapter
    conditions.py   condition language shared by rules and skill branches
    tbox.py         world model + gap detection + evolution proposals
    abox.py         situation assembly + preference learning
    rbox.py         rules, decisions, evidence graphs
    skills.py       skill computational graph (sequence / condition / loop)
    evolution.py    decision deltas, counterfactuals, back-testing, governance
    firewall.py     the firewall domain: seeds, ops, pipeline
sources/agents/firewall_cr_agent.py    conversational shell
examples/firewall_cr_demo.py           the whole loop, offline, in one run
```

## The four cognitive assets

| Asset | Holds | Learns |
|---|---|---|
| TBox | which concepts may exist | new concepts, properties, relations |
| ABox | what is currently true | new facts, user preferences, org state |
| RBox | what must happen when | new rules, conditions, exceptions |
| Skill | how the work is carried out | new steps, branches, loops, tools |

Each is versioned in Semantica and never mutated in place: an approved proposal
*derives* v1.1 from v1.0 with a changelog, so you can always ask why the agent
behaves differently this month than last.

## The loop

```
                ┌──────────────┐
                │    TBox      │
                │ World Model  │
                └──────┬───────┘
                       │
                       ▼
Ticket ───────► Situation ABox
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
    Personal ABox   Org ABox      Task ABox
          │            │             │
          └────────────┼─────────────┘
                       ▼
                     RBox
                       │
                       ▼
               Skill Execution
          sequence / condition / loop
                       │
                       ▼
                Evidence Graph
                       │
                       ▼
                Recommendation
                       │
                       ▼
                 Human Action
                       │
                       ▼
                   Feedback
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         ΔABox      ΔRBox       ΔSkill
            │          │          │
            └──────► ΔTBox ◄──────┘
                       │
                       ▼
                  New Version
```

Read as KSTAR: **K** (assets) conditions **S** (Situation ABox), **T/Skill**
decomposes and executes, **A** is the recommendation, **R** is the human action,
and reflection produces the deltas. Learning here is not fine-tuning; it is the
*verifiable evolution of cognitive assets*.

## Situation ABox

The ABox the agent reasons over is not a database read. For each ticket a
working subgraph is assembled across ontologies:

```
Personal ABox ----\
Org ABox ----------\
Infrastructure -----> Situation ABox
Task ABox ---------//
Ticket ABox ------//
```

`SituationBuilder` walks out from the ticket for a bounded number of hops,
keeps the source ontology of every fact in its provenance, and drops dangling
edges so the situation is self-contained and replayable. Rules then read it by
path - `destination.networkZone`, `requester.approvalAuthority`,
`source.networkZone.trustLevel` - and each match keeps the node it came from,
which is how the evidence graph gets built.

An unresolved path is a finding, not a gap to fill with a guess: it is reported
as an unresolved fact and the skill's retrieval loop tries to fill it from
Semantica.

## Rules and decisions

Rules are data (`Cmp` / `All` / `Any_` / `Not`), so they can be proposed,
serialised, back-tested and approved without anyone editing Python. Conflicts
resolve by *most restrictive wins*, which has a useful consequence: a learned
rule can only ever tighten an outcome, never silently loosen one.

Every decision carries an evidence graph:

```
Decision ─supportedBy→ Rule ─matchedOn→ Fact ─assertedOn→ Entity
```

## How a rule is learned

The worked example in `examples/firewall_cr_demo.py`:

1. R102 says PCI traffic *from outside the PCI zone* needs security review.
2. A vendor request arrives whose source is already inside the PCI zone, so
   nothing fires and the recommendation is AUTO_APPROVE.
3. The security engineer decides SECURITY_REVIEW: "external vendor access
   requires review regardless of network zone".
4. `CounterfactualAnalyzer` enumerates every fact the situation knows and asks
   of each: if R102 had also looked at this, would the human's decision follow
   *without changing any other case in history*?
5. Survivors are back-tested over the case archive. The winner amends the
   sub-condition that blocked the rule:

```
R102-v1.1:
IF destination.networkZone = 'PCI-Zone'
AND ( source.networkZone != 'PCI-Zone'
      OR requester.organizationType = 'ExternalVendor' )
THEN SECURITY_REVIEW
```

6. `GovernanceGate.approve` requires a named approver and a clean back-test
   (an override must state its reason) and produces RBox v1.1.

One case is not enough to generalise, and the design makes that visible: with a
single vendor case, `requester.role` and `requester.department` explain the
decision just as well as `organizationType` and all three are returned for a
human to choose between. After a second vendor case with a different role and a
different department, only `organizationType` fixes both, and it ranks first.
Diversity of evidence, not volume, is what earns a generalisation.

## The other three kinds of learning

- **ΔTBox** - the same case shows `organizationType` is a property the world
  model never declared. `OntologyGapDetector` proposes adding it; approving it
  yields TBox v1.1. New concepts are proposed the same way when facts of an
  undeclared type appear (for example `TemporaryAccessException`).
- **ΔABox** - a reviewer who keeps escalating production changes is stating a
  preference. `PreferenceLearner` accumulates signals, scores them
  (`support x agreement`, so contradictions lower confidence), and promotes them
  into the personal ontology above a threshold, with the sessions that produced
  them kept in provenance. R105 then fires - behaviour changed without any rule
  changing.
- **ΔSkill** - external vendor requests need a risk assessment the procedure
  never had. The patch inserts a `Branch` after the ownership check, producing
  skill v1.1. The rules were fine; the *computational graph* was wrong.

## Running it

```bash
python3 examples/firewall_cr_demo.py          # the whole loop, no LLM, no network
python3 -m unittest tests.test_semantica_firewall -v
python3 -m unittest tests.test_firewall_cr_agent -v   # needs the project runtime
```

Through the agent:

```
{"id": "FW-10293", "requester": "alice", "application": "PaymentApp",
 "source": "10.23.1.0/24", "destination": "DB-PROD-07", "port": 5432}

{"case_id": "FW-10293", "human_decision": "SECURITY_REVIEW", "reason": "..."}

approve rule-proposal-FW-10293-requester-organizationType
status
```

## Storage backends

`open_store()` returns the embedded property-graph backend unless `SEMANTICA_URL`
is set, in which case it returns `HttpSemanticaStore` (with `SEMANTICA_API_KEY`).
Both implement the same contract, so the layers above are unaware of which is in
use. The HTTP adapter's request shaping is unit-tested against an injected
session; it has not been exercised against a live Semantica server, and the
endpoint paths in its docstring are the contract it expects.

## Deliberate limits

- The counterfactual analyzer is a deterministic search over paths within two
  hops of the ticket, not causal inference. It proposes; it never concludes.
- Back-testing is only as good as the case archive; an empty archive makes every
  hypothesis look safe, which is why proposals are ranked but still handed to a
  human.
- `GovernanceGate` refuses an unnamed approver and an un-back-tested proposal by
  design. There is no code path in which the agent edits a live rule set.
