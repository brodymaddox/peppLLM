from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import random
import datetime
from finetuner import perform_fine_tuning

# Load documents from the RAG_DOCS folder
loader = DirectoryLoader('RAG_DOCS', glob='**/*.*', loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings()

# Create a Chroma vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Initialize the base Ollama LLM
base_llm = Ollama(model="llama3.1")

# Create a retrieval-based QA chain for the base model
base_qa_chain = RetrievalQA.from_chain_type(
    llm=base_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Prepare data for fine-tuning (unlabeled dataset)
fine_tuning_data = []
for doc in texts:
    fine_tuning_data.append({"text": doc.page_content})

# Save the fine-tuning data to a JSON file
fine_tuning_data_path = "fine_tuning_data.json"
with open(fine_tuning_data_path, "w") as f:
    json.dump(fine_tuning_data, f)

# Fine-tune the model
output_model = "fine_tuned_llama3.1"
perform_fine_tuning(fine_tuning_data_path, output_model)

# Load the fine-tuned model using Hugging Face Transformers
fine_tuned_model = AutoModelForCausalLM.from_pretrained(output_model)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(output_model)

# Function to query both models
def query_models(question: str):
    base_response = base_qa_chain.run(question)
    
    # Generate response from fine-tuned model
    inputs = fine_tuned_tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = fine_tuned_model.generate(**inputs, max_length=100)
    fine_tuned_response = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return base_response, fine_tuned_response

# Function to evaluate semantic similarity
def evaluate_similarity(response: str, reference: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    response_embedding = model.encode(response, convert_to_tensor=True)
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(response_embedding, reference_embedding).item()
    return similarity

# Load test questions from a JSON file
with open("test_questions.json", "r") as f:
    test_questions = json.load(f)

# Evaluate both models
base_scores = []
fine_tuned_scores = []
inter_model_scores = []

# Create a timestamp for the output file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"model_performance_{timestamp}.txt"

with open(output_filename, "w") as output_file:
    for test_item in test_questions:
        question = test_item["question"]
        ideal_response = test_item["ideal_response"]
        base_response, fine_tuned_response = query_models(question)
        
        base_ideal_similarity = evaluate_similarity(base_response, ideal_response)
        fine_tuned_ideal_similarity = evaluate_similarity(fine_tuned_response, ideal_response)
        inter_model_similarity = evaluate_similarity(base_response, fine_tuned_response)
        
        base_scores.append(base_ideal_similarity)
        fine_tuned_scores.append(fine_tuned_ideal_similarity)
        inter_model_scores.append(inter_model_similarity)
        
        output_file.write(f"Question: {question}\n")
        output_file.write(f"Ideal response: {ideal_response}\n")
        output_file.write(f"Base model response: {base_response}\n")
        output_file.write(f"Fine-tuned model response: {fine_tuned_response}\n")
        output_file.write(f"Base model similarity to ideal: {base_ideal_similarity}\n")
        output_file.write(f"Fine-tuned model similarity to ideal: {fine_tuned_ideal_similarity}\n")
        output_file.write(f"Inter-model similarity: {inter_model_similarity}\n")
        output_file.write("--------------------\n")
        
        print(f"Question: {question}")
        print(f"Ideal response: {ideal_response}")
        print(f"Base model response: {base_response}")
        print(f"Fine-tuned model response: {fine_tuned_response}")
        print(f"Base model similarity to ideal: {base_ideal_similarity}")
        print(f"Fine-tuned model similarity to ideal: {fine_tuned_ideal_similarity}")
        print(f"Inter-model similarity: {inter_model_similarity}")
        print("--------------------")

    # Calculate average scores
    avg_base_score = sum(base_scores) / len(base_scores)
    avg_fine_tuned_score = sum(fine_tuned_scores) / len(fine_tuned_scores)
    avg_inter_model_score = sum(inter_model_scores) / len(inter_model_scores)

    output_file.write(f"Average base model score: {avg_base_score}\n")
    output_file.write(f"Average fine-tuned model score: {avg_fine_tuned_score}\n")
    output_file.write(f"Average inter-model similarity score: {avg_inter_model_score}\n")

    print(f"Average base model score: {avg_base_score}")
    print(f"Average fine-tuned model score: {avg_fine_tuned_score}")
    print(f"Average inter-model similarity score: {avg_inter_model_score}")

print(f"Performance results saved to {output_filename}")