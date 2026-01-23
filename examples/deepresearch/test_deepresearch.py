import asyncio
from rllm.engine.rollout import OpenAIEngine
from deepresearch_agent import MultiTurnReactAgent
from deepresearch_tools import get_all_tools


async def main():
    # Setup rollout engine
    engine = OpenAIEngine(
        model="Qwen3-32B",
        api_key="QSTcad6c69c9d0f180ad1b8862e4b25d82b",
        base_url="http://redservingapi.devops.xiaohongshu.com/v1"
    )

    # Create agent with tools
    agent = MultiTurnReactAgent(
        rollout_engine=engine,
        tools=get_all_tools()
    )

    # Run a research task
    result = await agent.run(
        question="What is the reduced 12th dimensional Spin bordism of BG2?",
        answer="Z/2",  # Optional ground truth for evaluation
    )

    print(result)

    print(f"Prediction: {result['prediction']}")
    print(f"Rounds: {result['rounds']}")
    print(f"Time taken: {result['time_taken']}s")


if __name__ == "__main__":
    asyncio.run(main())
