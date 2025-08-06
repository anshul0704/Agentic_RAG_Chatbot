#from langchain.tools import tool
from langchain.agents import Tool

#from langchain.utilities import PythonREPL
from langchain_experimental.utilities import PythonREPL



calculator = Tool(
    name="calculator",
    func=PythonREPL().run,
    description="A calculator that can perform basic questions like addition, subtraction, multiplication, and division. "
)