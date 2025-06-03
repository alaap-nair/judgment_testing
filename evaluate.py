from dotenv import load_dotenv
load_dotenv() 
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer

client = JudgmentClient()  # reads JUDGMENT_API_KEY + JUDGMENT_ORG_ID

example = Example(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30-day full refund at no extra cost."
    ],
)

scorer = FaithfulnessScorer(threshold=0.5)

results = client.run_evaluation(
    examples=[example],
    scorers=[scorer],
    model="gpt-4.1",
)

print(results)
