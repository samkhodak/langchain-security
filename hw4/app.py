from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langsmith import Client
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.tools import tool
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import traceback
import os
import re

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"gensec-hw4"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()

gemini_llm = GoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
    }
)

class DocumentFilename(BaseModel):
    """
    This class enforces typechecking in the load_code_file tool.
    It extends the BaseModel class from Pydantic. 
    :param file_name: name of a file, including extension. Remove any previous path before the filename.
    :type TODO: str
    """
    file_name: str = Field(description="Should be a filename string with a suffix, such as code.py or code.cpp - Nothing else is accepted. ")
    @validator('file_name')
    def validate_filename(cls, value):
        file_name = value.replace("'", "")
        pattern = None
        try:
            pattern = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?\.[a-zA-Z0-9_-]+$", re.M)
        except Exception:
            traceback.print_exc()

        result = re.search(pattern, value)
        if not result:
            raise ValueError("Invalid filename.extension")
        return result.group()


@tool("load_code_file", args_schema=DocumentFilename, return_direct=False)
def load_code_file(file_name):
    """
    Useful for retrieving code from a document in the filesystem. Given a file name, the tool will retrieve the file 
    and copy the code inside of it, returning a string of code for the LLM to analyze. 
    """

    loader = GenericLoader.from_filesystem(
            path=f"./{file_name}",
            glob="*",
            suffixes=[".py", ".txt", ".cpp"],
            parser=LanguageParser(),
    )
    docs = loader.load()
    document_code = "\n\n\n".join([document.page_content for document in docs])

    prompt = PromptTemplate.from_template("""You are an intelligent AI code deobfuscation bot. Your directive is to take a piece of
        code and deobfuscate it, making it more understandable to the human programmer. Take each piece of deobfuscation step-by-step, so that
        the final result is a block of code that makes sense as a whole, and the purpose of the code is understandable. Improve any 
        potentially confusing variable names with better, self-documenting names.
        Your final answer MUST be in code format - output only a string of code with no backticks.
        Code content: {code_content}
    """)

    deobfusc_chain = ({"code_content":RunnablePassthrough()} | prompt | gemini_llm)
    result = deobfusc_chain.invoke(document_code)

    # Sometimes the LLM outputs the code with ```python [CODE]```. This trims the first and last line of the code to remove the backticks.
    # trimmed_result = "\n".join(result.split("\n")[1:-1])

    new_filename = f"deobfuscated_{file_name}"
    print(f"\n\nYour deobfuscated code has been saved to the following file: ./{new_filename}")
    with open(new_filename, "w") as file:
        file.write(result)

    return result
    






def main():
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions="""You are an agent that is used for helping the user view and deobfuscate code.
        Be as helpful as possible. If you are unable to produce an answer that is helpful to the user, say so.
        The user is allowed to look up information related to programming and deobfuscation ONLY. Deny them in any other case.
        Use AT MOST one call to load_code_file when it is needed.""")

    tools = load_tools(["serpapi"])
    tools.extend([load_code_file])

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

    print("Thanks for using the deobfuscator!")


if __name__ == "__main__":
    main()
