import torch
from transformers import RagTokenizer, RagSequenceForGeneration, RagRetriever

def main():
    
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")


    retriever = RagRetriever.from_pretrained(
        "facebook/rag-sequence-nq",
        index_name="custom",
        passages_path="data/passages",
        index_path="data/formula1_dpr.index"
    )

    
    model.set_retriever(retriever)

    
    question = "When was the first Formula 1 World Championship held?"


    inputs = tokenizer(question, return_tensors="pt")

   
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=5,
        num_return_sequences=1,
        max_length=100,
    )

    # Decode and print answer
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("Question:", question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
