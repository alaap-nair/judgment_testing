"""
batch_run.py – fires 20 GPT-Researcher jobs in parallel
Each job is fully traced & evaluated by Judgment.
"""
from dotenv import load_dotenv
load_dotenv() 
import asyncio, random, time
from judgeval.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer
from gpt_researcher import GPTResearcher
from openai import OpenAI

# ---------- tracing ----------
tracer = Tracer(project_name="GPT Researcher Load-Test")
wrap(OpenAI())                           # traces every OpenAI call

TOPICS = [
    "AI regulation timeline in the EU",
    "History of quantum supremacy claims",
    "Supply-chain risks for cobalt mining",
    "Impact of El Niño on global wheat prices",
    "Deepfake detection techniques 2025",
    "Cost drivers of small-modular reactors",
    "Ethics of brain–computer interfaces",
    "Nvidia vs AMD GPU market share 2024",
    "Drought-resistant maize genetics",
    "Quantum networking milestones",
    "Hydrogen fuel logistics in aviation",
    "Global lithium recycling startups",
    "Climate impact of cement alternatives",
    "Ocean iron fertilization research",
    "Privacy laws on facial recognition",
    "Roadmap for room-temperature superconductors",
    "Cybersecurity of EV charging stations",
    "VR therapy outcomes in PTSD",
    "Satellite mega-constellation debris risks",
    "AI chip export controls Asia"
]

@tracer.observe(span_type="agent")
async def run_research(query: str):
    researcher = GPTResearcher(query=query, report_type="research_report")
    await researcher.conduct_research()
    report = await researcher.write_report()

    tracer.async_evaluate(
        input=query,
        actual_output=report,
        scorers=[AnswerRelevancyScorer(threshold=0.7)],
        model="gpt-4o",
    )
    return report

async def main():
    t0 = time.time()
    # launch 20 jobs concurrently
    reports = await asyncio.gather(*[run_research(q) for q in TOPICS])
    print(f"Finished {len(reports)} reports in {time.time()-t0:0.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
