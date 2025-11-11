from dotenv import load_dotenv
from langsmith import Client
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

load_dotenv()

tools = [TavilySearch(time_range="month", topic="general")]
llm = ChatOllama(model="llama3.2")

client = Client()
react_prompt = client.pull_prompt("hwchase17/react")

system_prompt_str = react_prompt.format(**{var: "" for var in react_prompt.input_variables})

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt_str
)

def main():
    print("Hello from langchain-course!")

    user_input = "Search for 3 job postings for an AI engineer using LangChain in the Bay Area on LinkedIn and list their details"

    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    print("Result:", result.content if hasattr(result, "content") else result)

if __name__ == "__main__":
    main()
