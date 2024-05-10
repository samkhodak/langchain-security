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
    model="gemini-1.5-pro-latest",
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


def load_code(file_name) -> str:
    """
    Loads a file from current directory, parsing the file into code and returning it as a string.
    :param file_name: name of a file, including extension.
    :type file_name: str
    :return: string of code from loaded documents
    :rtype: str
    """
    loader = GenericLoader.from_filesystem(
            path=f"./{file_name}",
            glob="*",
            suffixes=[".py", ".txt", ".cpp"],
            parser=LanguageParser(),
    )

    docs = loader.load()
    if not docs:
        raise ValueError("The filename was not found. Try again.")

    document_code = "\n\n\n".join([document.page_content for document in docs])
    print("document code: " , type(document_code))
    return document_code


def save_edited_file(file_prefix, file_name, code_contents) -> None:
    """
    Saves a new file in current directory, given a filename, prefix, and file contents.
    Displays location of saved file.
    :param file_prefix: text to prepend to filename
    :type file_prefix: str
    :param file_name: name of original file
    :type file_name: str
    :param code_contents: string of code, formatted in a language
    :type code_contents: str
    """
    new_filepath = f"./{file_prefix}{file_name}"
    with open(new_filepath, "w") as file:
        file.write(code_contents)
    print(f"\n\nYour commented code has been saved to the following file: {new_filepath}\n\n")


@tool("deobfuscate_code", args_schema=DocumentFilename, return_direct=True)
def deobfuscate_code(file_name):
    """
    Useful for retrieving code from a document in the filesystem and deobfuscating the code. Given a 
    file name, the tool will retrieve the file and analyze the code inside of it, returning a string of 
    deobfuscated code for the user to use. When the code is returned, it has been successfully deobfuscated.
    :param file_name: filename without path, with extension.
    :type file_name: str
    :return: modified deobfuscated code
    :rtype: str
    """

    document_code = load_code(file_name)

    prompt = PromptTemplate.from_template("""You are an intelligent AI code deobfuscation bot. 
        Your directive is to take a piece of code and deobfuscate it, making it more understandable to 
        the human programmer. Take each piece of deobfuscation step-by-step, so that the final result is 
        a block of code that makes sense as a whole, and the purpose of the code is understandable. 
        Improve any potentially confusing variable names with better, self-documenting names. 
        Make sure to move imports or includes to the top of the code.
        Your final answer MUST be in code format - output only a string of code with no backticks.
        Code content: {code_content}
    """)

    deobfuscation_chain = ({"code_content":RunnablePassthrough()} | prompt | gemini_llm)
    final_code = deobfuscation_chain.invoke(document_code)

    save_edited_file("deobfuscated_", file_name, final_code)

    return final_code


@tool("comment_code", args_schema=DocumentFilename, return_direct=True)
def comment_code(file_name) -> str:
    """
    This tool will take a document of code in the filesystem and analyze it, adding comments
    in the process, and return an updated string of code that includes comments. 
    :param file_name: filename without path, with extension.
    :type file_name: str
    :return: modified code with comments
    :rtype: str
    """

    document_code = load_code(file_name)

    prompt = PromptTemplate.from_template("""You are an intelligent AI code commenting bot. 
        Your directive is to take in a section of code and add documentation comments to it to make it
        more understandable and well-documented. Make sure to keep imports or includes at the top of the code.
        Here are the rules to commenting:
        1. Your comments must be using the Google style for each programming language.
        3. You may only add comments to function declarations and class/enum declarations. No inline comments, and no constructor comments.
        4. Keep your comments concise, less than 45 words.
        5. Your final answer MUST be in code format - output only a string of code with NO BACKTICKS surrounding it. 
        Code content: {code_content}
    """)

    commenting_chain = ({"code_content":RunnablePassthrough()} | prompt | gemini_llm)
    final_code = commenting_chain.invoke(document_code)

    save_edited_file("commented_", file_name, final_code)

    return final_code






def main():
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions="""You are an agent that is used for helping the user deobfuscate and comment code.
        Be as helpful as possible. If you are unable to produce an answer that is helpful to the user, say so.
        The user is allowed to look up information related to programming and deobfuscation ONLY. Deny them in any other case.""")

    tools = load_tools(["serpapi"])
    tools.extend([deobfuscate_code, comment_code])

    gemini_agent = create_react_agent(gemini_llm, tools, prompt)
    gemini_executor = AgentExecutor(
            agent=gemini_agent, 
            tools=tools, 
            max_iterations=4, 
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

        except ValueError as v_error:
            print(f"\n\n{str(v_error)}")
        except Exception:
            traceback.print_exc()

    print("Thanks for using code-modifying agent!")


if __name__ == "__main__":
    main()
