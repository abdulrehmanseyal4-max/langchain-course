from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], run_id: str, **kwargs: Any
    ) -> None:
        print(f"\n🟢 LLM Run Started (run_id={run_id})")
        print(f"Prompt Sent:\n{prompts[0]}\n")

    def on_llm_end(self, response: LLMResult, run_id: str, **kwargs: Any) -> None:
        print(f"\n🔵 LLM Run Ended (run_id={run_id})")
        print(f"Response:\n{response.generations[0][0].text}\n")
