from judgeval import JudgmentClient
client = JudgmentClient()   # picks up env vars
client.export_traces(
    project_name="GPT Researcher Demo",
    output_path="gpt_researcher_traces.parquet",
    output_format="parquet",
)