from datasets import load_from_disk
from transformers import RagTokenizer, BartTokenizer, RagSequenceForGeneration, RagRetriever
import torch
from tqdm import tqdm
import evaluate

# -------------------------------
# 1. Load model, retriever, and tokenizers
# -------------------------------
model_dir = "rag_finetuned_model"
index_path = "data/formula1_dpr.index"
passages_path = "data/passages"

# Load tokenizer
tokenizer = RagTokenizer.from_pretrained(model_dir)
bart_tokenizer = BartTokenizer.from_pretrained(model_dir)

# Load retriever
retriever = RagRetriever.from_pretrained(
    model_dir,
    index_name="custom",
    index_path=index_path,
    passages_path=passages_path,
)

# Load model with retriever
model = RagSequenceForGeneration.from_pretrained(model_dir, retriever=retriever)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load test dataset
dataset = load_from_disk("data/qa/formula1_split")
test_dataset = dataset["test"]

rouge = evaluate.load("rouge")

def normalize_text(text):
    return text.lower().strip()

predictions = []
references = []

for example in tqdm(test_dataset, desc="Evaluating"):
    question = example["question"]
    answer = example["answer"]
    if isinstance(answer, list):
        answer = " / ".join(answer)

    # Prepare batch for RAG
    inputs = tokenizer([question], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=64
        )

    pred = bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    predictions.append(normalize_text(pred))
    references.append(normalize_text(answer))

# -------------------------------
# 3. Compute metrics
# -------------------------------
em_score = sum([p == r for p, r in zip(predictions, references)]) / len(predictions)
rouge_result = rouge.compute(predictions=predictions, references=references)

def f1_score(pred, ref):
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    common = set(pred_tokens) & set(ref_tokens)
    if not pred_tokens or not ref_tokens:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

f1_scores = [f1_score(p, r) for p, r in zip(predictions, references)]
f1_avg = sum(f1_scores) / len(f1_scores)

print("\n=== Evaluation Results ===")
print(f"Exact Match (EM): {em_score*100:.2f}%")
print(f"F1: {f1_avg*100:.2f}%")
print(f"ROUGE: {rouge_result}")
