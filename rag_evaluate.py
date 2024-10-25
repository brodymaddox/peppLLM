import os
import torch
import json
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer, util
from langchain.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

loader = DirectoryLoader("./documents", loader_cls=PyPDFLoader)  # Loading PDF documents
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents, embedding_model, persist_directory="./chroma_db")

pretrained_model_name = "gpt2"
pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

llm_pipeline = pipeline("text-generation", model=pretrained_model, tokenizer=pretrained_tokenizer, max_new_tokens=50)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

retriever = vector_store.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "{context}"
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

with open("questions.json", "r") as f:
    test_questions = json.load(f)

scores = []

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"rag_system_evaluation_{timestamp}.txt"

with open(output_filename, "w") as output_file:
    for test_item in test_questions:
        question = test_item["question"]
        ideal_response = test_item["ideal_response"]
        model_response = retrieval_qa.invoke({'input': question}, config={'max_new_tokens':50})['answer']
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
