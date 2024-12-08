import os
import torch
import json
import datetime
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer, util
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

def run_evaluation(model_name: str, test_questions_file: str, pdf_directories: list, is_multiple_choice: bool):
    raw_documents = []
    for pdf_directory in pdf_directories:
        loader = DirectoryLoader(pdf_directory, loader_cls=PyPDFLoader)
        raw_documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents, embedding_model, persist_directory="./chroma_db")

    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    pretrained_tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm_pipeline = pipeline("text-generation", model=pretrained_model, tokenizer=pretrained_tokenizer, max_new_tokens=150, device='cuda')
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    retriever = vector_store.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer **Do not include the context in your answer.**"
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "{context}"
    )

    if is_multiple_choice:
        system_prompt += (
            " Please respond with only the one letter (A, B, C, or D) "
            "corresponding to the correct answer."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_documents_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_qa = create_retrieval_chain(retriever, combine_documents_chain)

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
    output_filename = f"rag_system_evaluation_{model_name_short}_{test_file_name}_{timestamp}.txt"

    with open(output_filename, "w") as output_file:
        for test_item in tqdm(test_questions):
            question = test_item["Question"]
            ideal_response = test_item["Expected Response"]
            model_response = retrieval_qa.invoke({'input': question}, config={'max_new_tokens':150})['answer']
            
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
                output_file.write("Codec error with response")
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
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-3.1-8B", help="Name of the model to use")
    parser.add_argument("--test_questions_file", type=str, default="Similarity_Test_sets/full_similarity_contextual.json", help="Path to the test questions file")
    parser.add_argument("--pdf_directories", nargs='+', type=str, default=["./Contextual_Docs"], help="Paths to the directories containing PDF files")
    parser.add_argument("--multiple_choice", action="store_true", help="Flag to indicate if using multiple choice questions")
    args = parser.parse_args()

    run_evaluation(args.model_name, args.test_questions_file, args.pdf_directories, args.multiple_choice)
