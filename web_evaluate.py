import os
import torch
import json
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
from langchain.agents import initialize_agent, load_tools
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper

pretrained_model_name = "gpt2"
pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

llm_pipeline = pipeline("text-generation", model=pretrained_model, tokenizer=pretrained_tokenizer, max_new_tokens=50)

# Initialize Langchain's web browsing agent
tools = load_tools(["serpapi"])
llm = OpenAI(temperature=0)
web_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "{context}"
)

def evaluate_similarity(response: str, reference: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    response_embedding = model.encode(response, convert_to_tensor=True)
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(response_embedding, reference_embedding).item()
    return similarity

def retrieve_context_from_web(question: str):
    # Use the web agent to search the web and gather context relevant to the question
    search_results = web_agent.run(question)
    return search_results

with open("questions.json", "r") as f:
    test_questions = json.load(f)

scores = []

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"web_agent_evaluation_{timestamp}.txt"

with open(output_filename, "w") as output_file:
    for test_item in test_questions:
        question = test_item["question"]
        ideal_response = test_item["ideal_response"]
        
        # Retrieve context from the web
        context = retrieve_context_from_web(question)
        
        # Generate model response using the context retrieved from the web
        prompt = system_prompt.format(context=context) + "\nHuman: " + question + "\nAssistant: "
        model_response = llm_pipeline(prompt)[0]['generated_text'].split('Assistant:')[-1].strip()
        
        similarity_score = evaluate_similarity(model_response, ideal_response)
        scores.append(similarity_score)

        output_file.write(f"Question: {question}\n")
        output_file.write(f"Ideal response: {ideal_response}\n")
        output_file.write(f"Model response: {model_response}\n")
        output_file.write(f"Similarity score: {similarity_score}\n")
        output_file.write("--------------------\n")

        print(f"Question: {question}")
        print(f"Ideal response: {ideal_response}")
        print(f"Model response: {model_response}")
        print(f"Similarity score: {similarity_score}")
        print("--------------------")

    avg_score = sum(scores) / len(scores)
    output_file.write(f"Average similarity score: {avg_score}\n")
    print(f"Average similarity score: {avg_score}")

print(f"Evaluation results saved to {output_filename}")
