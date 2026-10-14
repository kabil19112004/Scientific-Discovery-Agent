from orchestration.agent_manager import AgentManager

m = AgentManager()
print(m.ingest("graph neural networks drug discovery", max_results=20))

review = m.review("How are graph neural networks used in drug discovery?")
print(review.to_markdown())

print(m.hypothesize(review).model_dump_json(indent=2))