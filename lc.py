""" For Testing API KEY """

from langchain_openai import OpenAI
llm = OpenAI(api_key="YOUR_KEY_HERE")
response = llm.invoke("hello world!")
print(response)