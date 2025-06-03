import pytest
from dotenv import load_dotenv
load_dotenv() 
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    AnswerRelevancyScorer,
    ToolOrderScorer,
)

# import the agent you want to test
from flow_demo import plan_trip

client = JudgmentClient()            # picks up your JUDGMENT_* env-vars

# ---------- Single-step assertion (content quality) ----------
def test_itinerary_relevancy():
    example = Example(
        input="Three-day Barcelona itinerary",
        # We’re about to inject a KNOWN bad answer, so we expect an assertion error
        actual_output="Totally irrelevant text",
    )
    with pytest.raises(AssertionError):
        client.assert_test(
            eval_run_name="itinerary_fail",
            examples=[example],
            scorers=[AnswerRelevancyScorer(threshold=0.8)],
            model="gpt-4o",       
        )

def test_tool_order():
    example = Example(
        input={"city": "Barcelona"},
        expected_tools=[
            {"tool_name": "get_weather"},
            {"tool_name": "search_restaurants"},
            {"tool_name": "search_museums"},
            {"tool_name": "compile_itinerary"},
        ],
    )

    scorer = ToolOrderScorer(exact_match=True)

    client.assert_test(
        eval_run_name="tool_order_ok",
        examples=[example],
        scorers=[scorer],
        function=lambda x: asyncio.run(plan_trip(x["city"])),
    )
