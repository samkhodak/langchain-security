from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langsmith import Client
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.tools import tool
import traceback
import os
import re

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"gensec-hw4"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()

class DocumentFilename(BaseModel):
    """
    This class enforces typechecking in the check_filename tool.
    It extends the BaseModel class from Pydantic. 
    :param file_name: name of a file, including extension. 
    :type TODO: str
    """
    file_name: str = Field(description="Should be a filename string with a suffix, such as code.py or code.cpp - Nothing else is accepted. ")
    @validator('file_name')
    def validate_filename(cls, value):
        pattern = None
        try:
            pattern = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?\.[a-zA-Z0-9_-]+$", re.M)
        except Exception:
            traceback.print_exc()

        result = re.search(pattern, value)
        if not result:
            raise ValueError("Invalid filename.extension")
        return result.group()


@tool("check_filename", args_schema=DocumentFilename, return_direct=False)
def check_filename(file_name):
    """
    Useful for checking the validity of a filename that the user passes in. Returns a filename that has been checked for validity..
    """
    return file_name




def main():
    gemini_llm = GoogleGenerativeAI(
        model="gemini-pro",
        temperature=0,
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
        }
    )
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions="""You are an agent that is used chiefly for helping the user deobfuscate code.
        Be as helpful as possible. If you are unable to produce an answer that is helpful to the user, say so.
        The user is allowed to look up information related to programming and deobfuscation ONLY. Deny them in any other case.""")

    tools = load_tools(["serpapi"])
    tools.extend([check_filename])

    gemini_agent = create_react_agent(gemini_llm, tools, prompt)
    gemini_executor = AgentExecutor(
            agent=gemini_agent, 
            tools=tools, 
            max_iterations=10, 
            early_stopping_method="generate", 
            verbose=True
    )

    while True:
        try:
            line = input("\n\nEnter query (\"exit\" to end) >>  ")
            if line and line != "exit": 
                result = gemini_executor.invoke({"input":line})
                print(f"\n\n{result.get('output')}")

            else:
                break

        except Exception:
            traceback.print_exc()
            break

    print("Thanks for using the deobfuscator!")


if __name__ == "__main__":
    main()
