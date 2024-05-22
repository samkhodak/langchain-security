from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langsmith import Client
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.tools import tool
import requests
import socket
import validators
import traceback
import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"gensec-hw6"
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

class IPv4Input(BaseModel):
    """ 
    For checking correctness of IPv4 address passed to tools.

    :param address: IPv4 address, without CIDR notation.
    :type address: str
    """
    address: str = Field(description="Should be an IP address such as 208.91.197.27 or 143.95.239.83, with no CIDR notation.")
    @validator('address')
    def is_ip_address(cls, value: str) -> str:
        if not validators.ip_address.ipv4(value):
            raise ValueError("Malformed IP address")
        return value


@tool("retrieve_DNS_name", args_schema=IPv4Input, return_direct=False)
def retrieve_DNS_name(ip_address):
    """
    Given an IPv4 address, returns DNS hostname associated with it

    :param ip_address: IPv4 address, without CIDR notation.
    :type ip_address: str
    :raises ValueError: Hostname is rejected if it's not able to be found.
    :return: DNS hostname for IP passed in
    :rtype: str
    """
    try:
        hostname, _, _ = socket.gethostbyaddr(ip_address)
        return hostname
    except socket.herror:
        raise ValueError("The hostname is not valid. Please enter an IPv4 address with no CIDR notation.")


@tool("ip_location_info", args_schema=IPv4Input, return_direct=False)
def ip_location_info(ip_address):
    """
    Get relevant location and organization information for an IP address.

    :param ip_address: IPv4 address, without CIDR notation.
    :type ip_address: str
    :return: dictionary of location info, including organization
    :rtype: dictionary of strings
    """
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_info = {
        'location': f"{response.get('city','N/A')}, {response.get('region','N/A')} - {response.get('country_name','N/A')} ({response.get('continent_code','N/A')})",
        'latitude': response.get('latitude', 'N/A'),
        'longitude': response.get('longitude', 'N/A'),
        'organization': response.get('org', 'N/A')
    }
    return location_info
    




def main():
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions="""You are an agent that is used for helping the user get information about IP addresses.
        Be as helpful as possible. If you are unable to produce an answer that is helpful to the user, say so.
        The user is allowed to look up information related to IP addresses ONLY. Deny them in any other case.
        Because your tools provide a lot of dense information, structure your response by separating each tool call's answer in a visually pleasing list.""")

    tools = load_tools(["serpapi"])
    tools.extend([retrieve_DNS_name, ip_location_info])

    gemini_agent = create_react_agent(gemini_llm, tools, prompt)
    gemini_executor = AgentExecutor(
            agent=gemini_agent, 
            tools=tools, 
            max_iterations=5, 
            early_stopping_method="generate", 
            verbose=True
    )

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

    return


if __name__ == "__main__":
    main()
