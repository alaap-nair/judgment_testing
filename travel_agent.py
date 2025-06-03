"""
A minimal but non-trivial agent that:
1.  Wraps OpenAI calls so every generation is traced.
2.  Uses @observe on helper “tools”.
3.  Fires an async evaluation inside the running trace.
Drop this file next to evaluate.py in the same project folder.
"""
from dotenv import load_dotenv
load_dotenv() 
from judgeval.tracer import Tracer, wrap           # tracing & observe :contentReference[oaicite:0]{index=0}
from judgeval.scorers import AnswerRelevancyScorer # built-in scorer :contentReference[oaicite:1]{index=1}
from openai import OpenAI

# ---------- instrumentation ----------
openai = wrap(OpenAI())                            # auto-captures latency/usage :contentReference[oaicite:2]{index=2}
judgment = Tracer(project_name="Test Travel Agent")  # shows up in the UI sidebar

# ---------- “tools” ----------
@judgment.observe(span_type="tool")
def weather(destination: str) -> str:
    """Fake weather lookup."""
    return f"The forecast in {destination} is sunny and 75 °F."

@judgment.observe(span_type="tool")
def flights(destination: str) -> str:
    """Fake flight search."""
    return f"Non-stop SFO → {destination} on UA123 for $350."

# ---------- agent ----------
@judgment.observe(span_type="agent")
def plan_trip(destination: str) -> str:
    context = f"{weather(destination)} {flights(destination)}"

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a high-energy travel assistant."},
            {"role": "user", "content": f"I want a one-week itinerary in {destination}."},
            {"role": "assistant", "content": context},
        ],
    )
    itinerary = response.choices[0].message.content

    # real-time quality guardrail
    judgment.async_evaluate(                      # runs inside the trace :contentReference[oaicite:3]{index=3}
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=f"One-week trip to {destination}",
        actual_output=itinerary,
        model="gpt-4o",
    )
    return itinerary

if __name__ == "__main__":
    print(plan_trip("Paris"))



