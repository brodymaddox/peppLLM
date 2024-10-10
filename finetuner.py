import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

def fine_tune_model(fine_tuning_data_path: str, output_model: str):
    # Load the fine-tuning data from JSON
    with open(fine_tuning_data_path, "r") as f:
        fine_tuning_data = json.load(f)

    # Convert the fine-tuning data to a Hugging Face Dataset
    dataset = Dataset.from_list(fine_tuning_data)

    # Load the pre-trained model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the actual model name
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_model,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        save_strategy="epoch",
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_model)

    print(f"Fine-tuning complete. The new model is saved in '{output_model}'")

    return model

# This function can be called from main.py to perform fine-tuning
def perform_fine_tuning(fine_tuning_data_path: str, output_model: str):
    return fine_tune_model(fine_tuning_data_path, output_model)