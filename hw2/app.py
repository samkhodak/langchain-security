from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain import hub
import traceback
import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"LangSmith Introduction"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()


def create_agent_executor(llm, tools, prompt):
    # We need an input_key for memory so the buffer doesn't confuse what passed-in input is from the user
    memory = ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

    return memory, executor




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
    gpt_llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    tools = load_tools(["terminal", "openweathermap-api"], allow_dangerous_tools=True)
    #tools.extend([tool1, tool2])


    base_prompt = hub.pull("langchain-ai/react-agent-template")
    final_prompt = base_prompt.partial(instructions="Use at most 10 tool calls. Prioritize use of tools beginning with custom_ if they are applicable to the user's question.")

    gemini_memory, gemini_executor = create_agent_executor(gemini_llm, tools, final_prompt)
    gpt_memory, gpt_executor = create_agent_executor(gpt_llm, tools, final_prompt)

    print("\nWelcome to my application. I am configured with these tools:\n")
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