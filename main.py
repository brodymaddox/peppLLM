from langchain import LLMChain, PromptTemplate
from langchain.llms import Ollama

# Define a prompt template to interact with Llama 3.1
prompt_template = """
You are a helpful assistant. Given the input below, provide an informative response.

Input: {input_text}
Response:
"""

# Create a prompt template instance
template = PromptTemplate(
    input_variables=["input_text"],
    template=prompt_template,
)

# Connect to the local Llama 3.1 instance via Ollama
llama_llm = Ollama(model="llama3.1")

# Create the Langchain LLMChain
llm_chain = LLMChain(
    llm=llama_llm,
    prompt=template,
)

# Define the text input to interact with the model
input_text = "What is the capital of France?"

# Run the chain and get the response
response = llm_chain.run(input_text)

# Print the output
print(response)