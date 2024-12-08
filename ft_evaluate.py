import os
import torch
import json
import datetime
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from tqdm import tqdm

def run_evaluation(model_name: str, test_questions_file: str, is_multiple_choice: bool):
    fine_tuned_model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model").to('cuda')
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

    llm_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    def query_model(question: str):
        if is_multiple_choice:
            prompt = "Please respond with only the one letter (A, B, C, or D) corresponding to the correct answer.\n\n"
            question = prompt + question
        inputs = fine_tuned_tokenizer(question, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = fine_tuned_model.generate(**inputs, max_new_tokens=150)
        response = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def evaluate_similarity(response: str, reference: str):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        response_embedding = model.encode(response, convert_to_tensor=True)
        reference_embedding = model.encode(reference, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(response_embedding, reference_embedding).item()
        return similarity

    with open(test_questions_file, "r") as f:
        test_questions = json.load(f)

    scores = []
    correct_answers = 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = model_name.split("/")[-1]
    test_file_name = os.path.splitext(os.path.basename(test_questions_file))[0]
    output_filename = f"fine_tuned_evaluation_{model_name_short}_{test_file_name}_{timestamp}.txt"

    with open(output_filename, "w") as output_file:
        for test_item in tqdm(test_questions):
            question = test_item["Question"]
            ideal_response = test_item["Expected Response"]
            model_response = query_model(question)
            
            if is_multiple_choice:
                is_correct = model_response.strip().upper() == ideal_response.strip().upper()
                correct_answers += int(is_correct)
                scores.append(int(is_correct))
                similarity_score = 1 if is_correct else 0
            else:
                similarity_score = evaluate_similarity(model_response, ideal_response)
                scores.append(similarity_score)

            output_file.write(f"Question: {question}\n")
            output_file.write(f"Ideal response: {ideal_response}\n")
            try:
                output_file.write(f"Model response: {model_response}\n")
            except:
                output_file.write("Error with model response codec")
            if is_multiple_choice:
                output_file.write(f"Correct: {is_correct}\n")
            else:
                output_file.write(f"Similarity score: {similarity_score}\n")
            output_file.write("--------------------\n")

        if is_multiple_choice:
            accuracy = correct_answers / len(test_questions)
            output_file.write(f"Accuracy: {accuracy}\n")
            return accuracy
        else:
            avg_score = sum(scores) / len(scores)
            output_file.write(f"Average similarity score: {avg_score}\n")
            return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuned model evaluation")
    parser.add_argument("--model_name", type=str, default="./fine_tuned_model", help="Path to the fine-tuned model")
    parser.add_argument("--test_questions_file", type=str, default="Similarity_Test_sets/full_similarity_contextual.json", help="Path to the test questions file")
    parser.add_argument("--multiple_choice", action="store_true", help="Flag to indicate if using multiple choice questions")
    args = parser.parse_args()

    run_evaluation(args.model_name, args.test_questions_file, args.multiple_choice)
