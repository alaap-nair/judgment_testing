# Multi‑Agent System Demo for Judgment Labs
"""
Self‑contained multi‑agent workflow **fully instrumented** for Judgeval.

Agents
------
1. **Researcher** – collects bullet‑point facts.
2. **Planner** – turns facts into a numbered plan.
3. **Critic** – spots flaws / hallucinations.
4. **Executor** – generates the final deliverable.

Run
~~~
```bash
export OPENAI_API_KEY="sk‑..."
export JUDGMENT_API_KEY="jl‑..."
python3 multiagent_system.py "Write a concise market brief on AR glasses in 2025."
```

Test
~~~~
```bash
python3 -m unittest multiagent_system.py
```

Dependencies
~~~~~~~~~~~~
- openai>=1.25  (Python SDK v1)
- judgeval>=0.40 (private index)
- python‑dotenv (optional)
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os, time, sys
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ────────────────────────────────────────────────────────────────────
# SDKs
# ────────────────────────────────────────────────────────────────────
try:
    import openai                                   # >=1.25.0
except ImportError:
    sys.exit("openai package not found → `pip install openai`. ")

try:
    from judgeval.common.tracer import Tracer, wrap # >=0.40
except ImportError:
    sys.exit("judgeval SDK missing → `pip install --extra-index-url https://pkg.judgmentlabs.ai/simple judgeval`. ")

# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    sys.exit("OPENAI_API_KEY not set in env.")

openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# **Project name must match what you select in the UI**
tracer = Tracer(project_name="MultiAgent", deep_tracing=True)

# Wrap an **instance** of the OpenAI client so every request is traced.
client = wrap(openai.OpenAI(api_key=openai_api_key))

# ────────────────────────────────────────────────────────────────────
# Base classes
# ────────────────────────────────────────────────────────────────────
class Agent:
    """Abstract LLM‑powered agent. Sub‑classes only override `build_prompt`."""

    def __init__(self, name: str):
        self.name = name
        self.memory: List[Dict[str, str]] = []  # local chat history

    def build_prompt(self, user_msg: str, shared: "Memory") -> List[Dict[str, str]]:
        raise NotImplementedError

    # Every agent call becomes a span
    @tracer.observe(span_type="agent")
    def __call__(self, user_msg: str, shared: "Memory") -> str:
        prompt = self.build_prompt(user_msg, shared)
        response = client.chat.completions.create(
            model=openai_model,
            messages=prompt,
            temperature=0.7,
        ).choices[0].message.content.strip()
        self.memory.append({"role": "assistant", "content": response})
        shared.append(self.name, response)
        return response

@dataclass
class Memory:
    """Lightweight shared blackboard."""

    store: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, author: str, content: str):
        self.store.append({"author": author, "content": content, "ts": time.time()})

    def last_n(self, n: int = 5) -> str:
        return "\n".join(f"[{m['author']}] {m['content']}" for m in self.store[-n:])

# ────────────────────────────────────────────────────────────────────
# Specialised agents
# ────────────────────────────────────────────────────────────────────
class Researcher(Agent):
    def build_prompt(self, user_msg: str, shared: Memory):
        sys_msg = (
            "You are Researcher, an expert at collecting citation‑ready facts. "
            "Return bullet points only—no opinion."
        )
        return [
            {"role": "system", "content": sys_msg},
            *self.memory,
            {"role": "user", "content": user_msg},
        ]

class Planner(Agent):
    def build_prompt(self, user_msg: str, shared: Memory):
        sys_msg = (
            "You are Planner. Using the researcher’s notes, craft a numbered "
            "execution plan that another agent can follow. Be specific."
        )
        context = f"Recent notes:\n{shared.last_n()}"
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": context},
        ]

class Critic(Agent):
    def build_prompt(self, user_msg: str, shared: Memory):
        sys_msg = (
            "You are Critic. Identify logical flaws, missing data, or hallucinations "
            "in the planner output. Respond with a JSON list of issues or []."
        )
        planner_output = shared.last_n(1)
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": planner_output},
        ]

class Executor(Agent):
    def build_prompt(self, user_msg: str, shared: Memory):
        sys_msg = (
            "You are Executor. Follow the approved plan and generate the final "
            "deliverable requested by the user. Include footer citations."
        )
        context = f"Plan & feedback:\n{shared.last_n()}"
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": context},
        ]

# ────────────────────────────────────────────────────────────────────
# Coordinator
# ────────────────────────────────────────────────────────────────────
class MultiAgentCoordinator:
    def __init__(self):
        self.memory = Memory()
        self.researcher = Researcher("Researcher")
        self.planner = Planner("Planner")
        self.critic = Critic("Critic")
        self.executor = Executor("Executor")

    @tracer.observe(span_type="function")  # root span
    def run(self, objective: str) -> str:
        # Phase 1: research
        self.researcher(objective, self.memory)
        # Phase 2: planning
        plan = self.planner(objective, self.memory)
        # Phase 3: critique loop (1‑pass)
        issues = self.critic(plan, self.memory)
        if "[]" not in issues:
            self.planner(f"Revise plan to fix issues: {issues}", self.memory)
        # Phase 4: execute
        final = self.executor(objective, self.memory)
        return final

# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 multiagent_system.py \"<objective string>\"")
        sys.exit(1)

    objective = sys.argv[1]
    output = MultiAgentCoordinator().run(objective)
    print("\n===== FINAL OUTPUT =====\n")
    print(output)

# ────────────────────────────────────────────────────────────────────
# Unit tests (Judgeval assert tests)
# ────────────────────────────────────────────────────────────────────
import unittest

class SanityTests(unittest.TestCase):
    def test_contains_keywords(self):
        obj = "Quantum Computing Overview"
        result = MultiAgentCoordinator().run(obj)
        self.assertIn("quantum", result.lower())

    def test_length_reasonable(self):
        obj = "Short memo on GPU trends"
        result = MultiAgentCoordinator().run(obj)
        self.assertLess(len(result.split()), 400)

if __name__ == "__test__":  # avoid auto‑discover
    unittest.main()
