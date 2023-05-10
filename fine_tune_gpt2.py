from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Hyperparameters
embedding_dim = 256
hidden_dim = 512
num_layers = 2
dropout_prob = 0.5
batch_size = 64
seq_len = 32
epochs = 10

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=seq_len)

def process_data_gpt2(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    examples = text.split('\n')
    return [{"text": example} for example in examples]

train_file_path = 'kids_stories_large.txt'
train_examples = process_data_gpt2(train_file_path, tokenizer)
# train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file_path, block_size=seq_len)
train_dataset = load_dataset("text", data_files=train_file_path)["train"]
train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    save_steps=10_000,
    save_total_limit=2,
    save_strategy="steps",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Save the model after training
trainer.save_model("./results")

