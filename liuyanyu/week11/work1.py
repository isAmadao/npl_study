import os
import asyncio
from pydantic import BaseModel
from typing import Optional
from agents import Agent,  Runner

from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


sentimental_analysis_agent = Agent(
  name='sentimental analysis agent',
  model='gpt-4o-mini',
  instructions="""You are an expert in sentiment analysis.
    Classify the sentiment of the given text as: Positive, Negative, or Neutral.
    Provide the classification result and briefly explain your reasoning.""",

)

entity_recognition_agent = Agent(
  name='entity_recognition_agent',
  model='gpt-4o-mini',
  instructions="""You are an expert in named entity recognition.
    Extract all named entities from the given text, including:
    - Person names (PER)
    - Locations (LOC)
    - Organizations (ORG)
    - Dates and times (TIME)
    - Other entities
    Present the results in a structured format.""",
)

agent = Agent(
  name="agent",
  model='gpt-4o-mini',
  instructions="""You are an intelligent routing assistant. Understand the user's intent and delegate to the right specialist agent.

  Routing rules:
    - User wants to analyze sentiment, emotion, or attitude -> hand off to sentiment_analysis_agent
    - User wants to extract entities, identify names / locations / organizations -> hand off to entity_recognition_agent
    """,

  handoffs=[sentimental_analysis_agent, entity_recognition_agent]

)

async def main():
    print("=== Multi-Agent System ===")
    print("Capabilities: Sentiment Classification | Named Entity Recognition")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        print("\nProcessing...\n")
        result = await Runner.run(agent, user_input)
        print(f"Result: {result.final_output}\n")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
