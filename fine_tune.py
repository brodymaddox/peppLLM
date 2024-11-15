import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, LlamaTokenizer, LlamaForCausalLM, DataCollatorWithPadding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
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

pdf_directory = "./documents"
pdf_texts = []

for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        pdf_texts.extend([page.page_content for page in document])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

texts = text_splitter.split_text("\n".join(pdf_texts))

model_name = "meta-llama/Llama-3.1-8B"

lora_config = LoraConfig(
    r=8,  # Low-rank matrix rank (you can adjust this)
    lora_alpha=32,  # Scaling factor for LoRA matrices
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Define which layers should be adapted (query and value projections in transformers)
    bias="none"  # Typically no bias for LoRA modules
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
