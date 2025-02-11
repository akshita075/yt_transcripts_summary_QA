import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

# ✅ Load TinyLlama model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ Load dataset from JSONL file
dataset = load_dataset("json", data_files="fine_tuning_dataset.jsonl")["train"]

# ✅ Tokenize the dataset
def tokenize_function(examples):
    tokenized_output = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()  # ✅ Ensure labels are included
    return tokenized_output

dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])

# ✅ Split dataset into train & eval
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ✅ Data Collator for Padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ Prepare Training Arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_youtube_model",
    eval_strategy="epoch",  # ✅ Updated from `evaluation_strategy` to `eval_strategy`
    save_strategy="epoch",
    per_device_train_batch_size=2,  # Reduce if RAM is low
    per_device_eval_batch_size=2,  
    num_train_epochs=3,  
    logging_dir="./logs",
    save_total_limit=2,  # Keep last 2 checkpoints only
    load_best_model_at_end=True,  # Use best model
)

# ✅ Custom Trainer with Fixed `compute_loss`
from torch.nn import CrossEntropyLoss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # ✅ Accept additional arguments
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # ✅ Compute loss manually
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # ✅ Updated from `tokenizer` to `processing_class`
    data_collator=data_collator  
)

# ✅ Start Fine-Tuning
print("🚀 Fine-Tuning TinyLlama on YouTube Transcripts...")
trainer.train()

# ✅ Save the fine-tuned model
model.save_pretrained("./fine_tuned_youtube_model")
tokenizer.save_pretrained("./fine_tuned_youtube_model")

print("✅ Fine-Tuning Completed! Model saved to `fine_tuned_youtube_model`.")