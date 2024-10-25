import os
import torch
import json
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

fine_tuned_model_path = "./fine_tuned_model"
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

llm_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

def query_model(question: str):
    inputs = fine_tuned_tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = fine_tuned_model.generate(**inputs, max_length=100)
    response = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def evaluate_similarity(response: str, reference: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    response_embedding = model.encode(response, convert_to_tensor=True)
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(response_embedding, reference_embedding).item()
    return similarity

with open("questions.json", "r") as f:
    test_questions = json.load(f)

scores = []

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"fine_tuned_evaluation_{timestamp}.txt"

with open(output_filename, "w") as output_file:
    for test_item in test_questions:
        question = test_item["question"]
        ideal_response = test_item["ideal_response"]
        model_response = query_model(question)
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

    # Calculate average score
    avg_score = sum(scores) / len(scores)
    output_file.write(f"Average similarity score: {avg_score}\n")
    print(f"Average similarity score: {avg_score}")

print(f"Evaluation results saved to {output_filename}")
