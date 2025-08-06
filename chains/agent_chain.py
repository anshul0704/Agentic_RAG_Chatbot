from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from rag_chain import rag_tool
from tools.calculator_tool import calculator
from tools.search_tools import search


llm= Ollama(model="mistral")

tools= [rag_tool, calculator, search]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_agent(query):
    return agent_executor.run(query)