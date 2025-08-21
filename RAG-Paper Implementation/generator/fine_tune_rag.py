from datasets import load_from_disk
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    Seq2SeqTrainingArguments,
    BartTokenizer,
    Seq2SeqTrainer
)
import torch

# -------------------------------
# 1. Load dataset
# -------------------------------
dataset = load_from_disk("data/qa/formula1_split")
print(dataset)

# -------------------------------
# 2. Load model + tokenizers
# -------------------------------
model_name = "facebook/rag-sequence-nq"
tokenizer = RagTokenizer.from_pretrained(model_name)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")  # for labels

retriever = RagRetriever.from_pretrained(
    model_name,
    index_name="custom",
    passages_path="data/passages",
    index_path="data/formula1_dpr.index"
)

model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)

# -------------------------------
# 3. Preprocessing function
# -------------------------------
def preprocess_function(examples):
    inputs = examples["question"]

    # Join multiple answers into a single string
    targets = [" / ".join(ans) if isinstance(ans, list) else str(ans)
               for ans in examples["answer"]]

    # Tokenize questions (inputs)
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Tokenize answers (labels)
    labels = bart_tokenizer(
        targets,
        truncation=True,
        padding="max_length",
        max_length=64
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

# -------------------------------
# 4. Tokenize dataset
# -------------------------------
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# -------------------------------
# 5. Training arguments
# -------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="rag_finetuned",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",         
    remove_unused_columns=False,   
)

# -------------------------------
# 6. Custom Trainer to fix non-scalar loss
# -------------------------------
class RagSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        # Extract loss safely
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # Ensure scalar loss
        if loss is not None and loss.dim() > 0:
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

trainer = RagSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# -------------------------------
# 7. Train and save
# -------------------------------
trainer.train()
trainer.save_model("rag_finetuned_model")
tokenizer.save_pretrained("rag_finetuned_model")
bart_tokenizer.save_pretrained("rag_finetuned_model")

print("Fine-tuned model saved to rag_finetuned_model/")
