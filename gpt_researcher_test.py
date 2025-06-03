from dotenv import load_dotenv
load_dotenv() 
import asyncio
from judgeval.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer
from gpt_researcher import GPTResearcher
from openai import OpenAI

# --- trace every OpenAI call ---
tracer = Tracer(project_name="GPT Researcher Demo")
wrap(OpenAI())                      # wrap() just needs the instance; we don’t use the return value

@tracer.observe(span_type="agent")
async def run_research(query: str):
    # ctor *does* take query + report_type  :contentReference[oaicite:0]{index=0}
    researcher = GPTResearcher(query=query, report_type="research_report")

    await researcher.conduct_research()        # pulls pages, runs crawler
    report = await researcher.write_report()   # stitches final report

    tracer.async_evaluate(                     # guardrail in the same trace
        input=query,
        actual_output=report,
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        model="gpt-4o",
    )
    return report

if __name__ == "__main__":
    print(asyncio.run(run_research("Why is Nvidia stock outperforming AMD?")))
