"""
flow_demo.py  –  deliberate flowchart test for Judgment Labs
Itinerary builder that calls four tools; two run concurrently.
"""
from dotenv import load_dotenv
load_dotenv()               

import asyncio, random, textwrap
from judgeval.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer
from openai import OpenAI

# ── tracing setup ──────────────────────────────────────────────────────────────
tracer = Tracer(project_name="Flowchart Demo", deep_tracing=True)
wrap(OpenAI())                                   
openai = OpenAI()                                

# ── tool 1 ─────────────────────────────────────────────────────────────────────
@tracer.observe(span_type="tool")
def get_weather(city: str) -> str:
    forecast = random.choice(["sunny", "rainy", "cloudy"])
    return f"The forecast for {city} next week is {forecast}."

# ── tool 2 (async) ────────────────────────────────────────────────────────────
@tracer.observe(span_type="tool")
async def search_restaurants(city: str) -> str:
    resp = await asyncio.to_thread(
        lambda: openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a food critic."},
                {"role": "user", "content": f"Top 3 must-eat restaurants in {city}."},
            ],
        )
    )
    return resp.choices[0].message.content.strip()

# ── tool 3 (async) ────────────────────────────────────────────────────────────
@tracer.observe(span_type="tool")
async def search_museums(city: str) -> str:
    resp = await asyncio.to_thread(
        lambda: openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a travel curator."},
                {"role": "user", "content": f"Three must-see museums in {city}."},
            ],
        )
    )
    return resp.choices[0].message.content.strip()

# ── tool 4 ─────────────────────────────────────────────────────────────────────
@tracer.observe(span_type="tool")
def compile_itinerary(city: str, weather: str, food: str, culture: str) -> str:
    return textwrap.dedent(f"""
        **{city} – 3-Day Itinerary**

        Weather snapshot: {weather}

        • Day 1 – Morning market crawl, lunch at {food.splitlines()[0]}
        • Day 2 – Spend afternoon at {culture.splitlines()[0]}
        • Day 3 – Free day, evening food tour

        Enjoy your trip!
    """)

# ── root agent ────────────────────────────────────────────────────────────────
@tracer.observe(span_type="agent")
async def plan_trip(city: str) -> str:
    weather = get_weather(city)                             # sync call → span 1

    # run two tools in parallel → two sibling spans
    food, culture = await asyncio.gather(
        search_restaurants(city),                           # span 2
        search_museums(city),                               # span 3
    )

    itinerary = compile_itinerary(city, weather, food, culture)  # span 4

    # live guard-rail (shows on right-hand panel)
    tracer.async_evaluate(
        input=f"Itinerary for {city}",
        actual_output=itinerary,
        scorers=[AnswerRelevancyScorer(threshold=0.6)],
        model="gpt-4o",
    )
    return itinerary

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(plan_trip("Barcelona"))
