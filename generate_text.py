import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch

import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch

def generate_text(prompt, model, tokenizer, max_length=500, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True)
    
    generated_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=0.8,
    )

    generated_texts = []
    for generated_sequence in generated_sequences:
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        generated_texts.append(text)

    return generated_texts

# Load the fine-tuned model from the output directory
model_path = "./results"
model = GPT2LMHeadModel.from_pretrained(model_path)
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Get user inputs
prompt = input("Enter a prompt: ")
max_length = int(input("Enter the max length for generated text: "))

# Generate text
generated_texts = generate_text(prompt, model, tokenizer, max_length=max_length)

# Print generated text
for idx, text in enumerate(generated_texts):
    print(f"Generated Text {idx + 1}:")
    print(text)
    print("\n")

