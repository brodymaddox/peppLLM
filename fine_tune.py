import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, LlamaTokenizer, LlamaForCausalLM, DataCollatorWithPadding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }

def run_fine_tuning(model_name: str, pdf_directories: list):
    pdf_texts = []

    for pdf_directory in pdf_directories:
        loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        pdf_texts.extend([page.page_content for document in documents for page in document])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    texts = text_splitter.split_text("\n".join(pdf_texts))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    model = get_peft_model(model, lora_config)

    tokenized_texts = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=200,
        fp16=True,
    )

    dataset = TextDataset(tokenized_texts)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class CustomTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch):
            inputs['labels'] = inputs['input_ids']
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            return super().training_step(model, inputs,num_items_in_batch)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    print("Model fine-tuning complete and saved locally.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", help="Name of the base model to use")
    parser.add_argument("--pdf_directories", nargs='+', type=str, default=["./Contextual_Docs"], help="Paths to the directories containing PDF files")
    args = parser.parse_args()

    run_fine_tuning(args.model_name, args.pdf_directories)
