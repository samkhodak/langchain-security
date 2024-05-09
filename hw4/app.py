from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langsmith import Client
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.tools import tool
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
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
    This class enforces typechecking for a provided file name. 
    It extends the BaseModel class from Pydantic. 
    :param file_name: name of a file, including extension. Remove any previous path before the filename.
    :type file_name: str
    """
    file_name: str = Field(description="Should be a filename string with a suffix, such as code.py or code.cpp - Nothing else is accepted. ")
    @validator('file_name')
    def validate_filename(cls, value):
        try:
            # Remove potential quotes before checking filename.
            file_name = value.replace("'", "").replace("\"", "")
            # This RegEx pattern should only accept filenames with a dot extension (e.g., code.cpp, app.py, a.txt)
            pattern = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?\.[a-zA-Z0-9_-]+$", re.M)
            result = re.search(pattern, file_name)
        except Exception:
            traceback.print_exc()

        if not result:
            raise ValueError("Invalid filename.extension")
        return result.group()


@tool("deobfuscate_code", args_schema=DocumentFilename, return_direct=False)
def deobfuscate_code(file_name):
    """
    Useful for retrieving code from a document in the filesystem and deobfuscating the code. Given a 
    file name, the tool will retrieve the file and analyze the code inside of it, returning a string of 
    deobfuscated code for the user to use. When the code is returned, it has been successfully deobfuscated.
    """

    loader = GenericLoader.from_filesystem(
            path=f"./{file_name}",
            glob="*",
            suffixes=[".py", ".txt", ".cpp"],
            parser=LanguageParser(),
    )

    docs = loader.load()
    document_code = "\n\n\n".join([document.page_content for document in docs])

    prompt = PromptTemplate.from_template("""You are an intelligent AI code deobfuscation bot. 
        Your directive is to take a piece of code and deobfuscate it, making it more understandable to 
        the human programmer. Take each piece of deobfuscation step-by-step, so that the final result is 
        a block of code that makes sense as a whole, and the purpose of the code is understandable. 
        Improve any potentially confusing variable names with better, self-documenting names. 
        Your final answer MUST be in code format - output only a string of code with no backticks.
        Code content: {code_content}
    """)

    deobfuscation_chain = ({"code_content":RunnablePassthrough()} | prompt | gemini_llm)
    result = deobfuscation_chain.invoke(document_code)

    new_filename = f"deobfuscated_{file_name}"
    with open(new_filename, "w") as file:
        file.write(result)
    print(f"\n\nYour deobfuscated code has been saved to the following file: ./{new_filename}\n\n")

    return result







def main():
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions="""You are an agent that is used for helping the user view and deobfuscate code.
        Be as helpful as possible. If you are unable to produce an answer that is helpful to the user, say so.
        The user is allowed to look up information related to programming and deobfuscation ONLY. Deny them in any other case.
        Use AT MOST one call to load_code_file when it is needed.""")

    tools = load_tools(["serpapi"])
    tools.extend([deobfuscate_code])

    gemini_agent = create_react_agent(gemini_llm, tools, prompt)
    gemini_executor = AgentExecutor(
            agent=gemini_agent, 
            tools=tools, 
            max_iterations=10, 
            early_stopping_method="generate", 
            verbose=True
    )

    print("\n\nWelcome to the Code-Modifying Agent. The agent has access to these tools:\n")
    for tool in gemini_executor.tools:
        print(f"\n{tool.name}: \n\n{tool.description}")

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
