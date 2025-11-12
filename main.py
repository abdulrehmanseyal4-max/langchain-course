from dotenv import load_dotenv
from langsmith import Client
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearchResults(max_results=5)]
llm = ChatOllama(model="llama3.2")
structured_llm = llm.with_structured_output(AgentResponse)

client = Client()

format_instructions = ""
system_prompt_str = REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS.format(
    tools="{tools}",
    tool_names="{tool_names}",
    input="{input}",
    agent_scratchpad="{agent_scratchpad}",
    format_instructions=format_instructions
)

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt_str)
extract_output = RunnableLambda(lambda x: x["messages"][-1].content)
chain = agent | extract_output | structured_llm


def main():
    print("Hello from langchain-course!")

    user_input = "Search for 3 job postings for an AI engineer using LangChain in the Bay Area on LinkedIn and list their details"

    result = chain.invoke({"messages": [HumanMessage(content=user_input)]})

    print("Result:", result.content if hasattr(result, "content") else result)


if __name__ == "__main__":
    main()
