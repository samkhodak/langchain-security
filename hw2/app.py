from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langsmith import Client
from langchain import hub
from textwrap import dedent
import traceback
import json
import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"gensec-hw2"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()


class ProjectNameInput(BaseModel):
    project_name: str = Field(description="Should be a string of text of the user's choice for an existing project name.")


@tool("project_token_info", args_schema=ProjectNameInput, return_direct=False)
def project_token_info(project_name):
    """
    This tool takes a project name and returns a string containing info about 
    the tokens used for that specific project in LangSmith.
    """
    results = client.read_project(project_name=project_name)
    run_count = results.run_count
    prompt_tokens = results.prompt_tokens
    completion_tokens = results.completion_tokens
    token_info = dict(
        run_count=run_count, 
        prompt_tokens_used=prompt_tokens,
        completion_tokens_generated=completion_tokens,
    )
    return json.dumps(token_info)


def create_agent_executor(llm, tools, prompt):
    # We need an input_key for memory so the buffer doesn't confuse what passed-in input is from the user
    memory = ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            memory=memory, 
            max_iterations=10, 
            early_stopping_method="generate", 
            verbose=True
    )
    return memory, executor




def main():
    gemini_llm = GoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0,
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
        }
    )
    gpt_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)


    tools = load_tools(["terminal", "openweathermap-api"], allow_dangerous_tools=True)
    tools.extend([project_token_info])


    base_prompt = hub.pull("langchain-ai/react-agent-template")
    final_prompt = base_prompt.partial(instructions="""When you use tools, you are allowed to use the response to craft a
        more specific answer. Your output to the user does not have to be the exact output of the tool.
        For the openweathermap-api tool, make sure your input corrects any spelling mistakes of location names.
        Some examples of good inputs: Portland, Oregon  
        London
        Examples of bad inputs: Portland, OR
        L0ndon
        For the project_token_info tool, make sure the string passed in doesn't have any extra characters
        surrounding the project name the user gives. You are free to use the response from the tool to craft
        a more human response.""")


    gemini_memory, gemini_executor = create_agent_executor(gemini_llm, tools, final_prompt)
    gpt_memory, gpt_executor = create_agent_executor(gpt_llm, tools, final_prompt)


    print("\nWelcome to my Gemini and GPT agents. The agent has access to these tools:\n")
    for tool in gemini_executor.tools:
        print(f'{tool.name}:  {tool.description}')


    while True:
        try:
            line = input("\n\nEnter query (\"exit\" to end) >>  ")
            if line and line != "exit": 
                print("\n\nGemini's answer: \n")
                result = gemini_executor.invoke({"input":line, "chat_history":gemini_memory})
                print(f"\n\n{result.get('output')}")

                print("\n\nGPT's answer: \n")
                result = gpt_executor.invoke({"input":line, "chat_history":gpt_memory})
                print(f"\n\n{result.get('output')}")

            else:
                break

        except Exception:
            traceback.print_exc()
            break
    
    print("\nThanks for using the Agent. Have a nice day!\n")




if __name__ == "__main__":
    main()
