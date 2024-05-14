from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
import traceback
import os
import re


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"gensec-hw5"
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
gpt_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# Maximum tokens allowed for a prompt, otherwise the chain call will be rejected.
MAX_PROMPT_TOKENS = 500



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
    return document_code


def main():

    gemini_chain = gemini_llm | StrOutputParser()
    gpt_chain = gpt_llm | StrOutputParser()


    while True:
        try:
            line = input("\n\nEnter query (\"exit\" to end) >>  ")
            if line and line != "exit": 
                document = load_code(line)
                if (gemini_llm.get_num_tokens(document) > MAX_PROMPT_TOKENS):
                    raise RuntimeError(f"The document you are trying to check is too large for the {MAX_PROMPT_TOKENS} token limit.")
                result = gemini_chain.invoke(document)
                print(result)

            else:
                break

        except ValueError as v_error:
            print(f"\n\n{str(v_error)}")
        except Exception:
            traceback.print_exc()


if __name__=="__main__":
    main()
