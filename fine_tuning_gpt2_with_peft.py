
# Install required packages
!pip install transformers datasets torch bitsandbytes-cuda111 peft # Replace 'cuda111' with your CUDA version

# Import Libraries
from transformers import GPT2Tokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from datasets import load_dataset, load_metric
import numpy as np
import torch
from torch.utils.data import DataLoader

# Load and Evaluate a Pre-trained Model

## a. Load Model and Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

## b. Load and Preprocess IMDb Dataset
dataset = load_dataset("imdb", split='train[:1%]')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': [label - 1 for label in examples['label']]})

## c. Evaluate the Pretrained Model
metric = load_metric("accuracy")

def evaluate_model(model, data_loader):
    model.eval()
    total_acc = 0
    for batch in data_loader:
        inputs = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = (predictions == labels).float().mean()
        total_acc += acc.item()
    return total_acc / len(data_loader)

data_loader = DataLoader(tokenized_dataset, batch_size=8)
pretrained_model_acc = evaluate_model(model, data_loader)
print("Accuracy of Pretrained Model:", pretrained_model_acc)

# Perform Parameter-Efficient Fine-Tuning with LoRA

## a. Creating a LoRA Config
lora_config = LoraConfig()

## b. Convert to a PEFT Model
peft_model = get_peft_model(model, lora_config)

## c. Train the PEFT Model
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

## d. Save the Trained Model
peft_model.save_pretrained("./peft_model")

# Perform Inference Using the Fine-Tuned Model and Compare Performance
fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained("./peft_model")
fine_tuned_model_acc = evaluate_model(fine_tuned_model, data_loader)
print("Fine-Tuned Model Accuracy:", fine_tuned_model_acc)
print("Pretrained Model Accuracy:", pretrained_model_acc)
