import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import pickle
from collections import Counter
import re
from torch.optim.lr_scheduler import StepLR
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embedding_dim = 256
hidden_dim = 512
num_layers = 2
dropout_prob = 0.5
batch_size = 16
seq_len = 64
epochs = 50

# # Original Hyperparameters
# embedding_dim = 256
# hidden_dim = 512
# num_layers = 2
# dropout_prob = 0.5
# batch_size = 64
# seq_len = 32
# epochs = 10

def train_bpe_tokenizer(file_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=300000,
        min_frequency=2,
        special_tokens=["<pad>", "<sos>", "<eos>", "<title>", "<eotitle>"]
    )
    tokenizer.train(files=[file_path], trainer=trainer)
    
    return tokenizer

# Load and preprocess dataset
# tokenizer = get_tokenizer('basic_english') # Removed old tokenizer
file_path = 'kids_stories_new.txt'
val_file_path = 'kids_stories_val_new.txt'
test_file_path = 'kids_stories_test_new.txt'
bpe_tokenizer = train_bpe_tokenizer(file_path) # Added new tokenizer
bpe_tokenizer.save("bpe_tokenizer.json")
bpe_tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# def process_data(file_path, tokenizer):
#     with open(file_path, 'r') as f:
#         text = f.read()

#     stories = text.split('\n\n')
#     filtered_stories = [story for story in stories if 50 < len(story) < 50000]  # Adjust character length constraint

#     tokenized_stories = []
#     for story in filtered_stories:
#         lines = story.split('\n')
#         title, content = lines[0], ' '.join(lines[1:])
#         tokenized_title = ['<title>'] + tokenizer(title) + ['<eotitle>']
#         tokenized_content = ['<sos>'] + tokenizer(content) + ['<eos>']
#         tokenized_stories.append(tokenized_title + tokenized_content)

#     return tokenized_stories

def process_data_bpe(file_path, tokenizer):
    with open(file_path, 'r') as f:
        text = f.read()

    stories = text.split('\n\n')
    filtered_stories = [story for story in stories if 50 < len(story) < 50000]

    tokenized_stories = []
    for story in filtered_stories:
        lines = story.split('\n')
        title, content = lines[0], ' '.join(lines[1:])
        tokenized_title = [1] + tokenizer.encode(title).ids + [4]  # Use tokenizer.encode().ids instead of tokenizer()
        tokenized_content = [2] + tokenizer.encode(content).ids + [3]
        tokenized_stories.append(tokenized_title + tokenized_content)

    return tokenized_stories

# tokenized_train_stories = process_data(file_path, tokenizer) # Added old tokenizer
tokenized_train_stories = process_data_bpe(file_path, bpe_tokenizer) # Added new tokenizer
# vocab = build_vocab_from_iterator(tokenized_train_stories, specials=["<unk>", "<pad>", "<sos>", "<eos>", "<title>","<eotitle>"])
# vocab.set_default_index(vocab["<unk>"])
vocab_size = bpe_tokenizer.get_vocab_size()

# def data_to_tensor(tokenized_stories, vocab, seq_len):
#     tensor_data = []
#     for story in tokenized_stories:
#         story_tensor = torch.tensor([vocab[token] for token in story], dtype=torch.long)
#         for i in range(0, len(story_tensor) - seq_len, seq_len):
#             tensor_data.append(story_tensor[i:i+seq_len])
#     return torch.stack(tensor_data)

# train_data = data_to_tensor(tokenized_train_stories, vocab, seq_len)
# train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

def data_to_tensor_bpe(tokenized_stories, tokenizer, seq_len):
    tensor_data = []
    for story in tokenized_stories:
        story_tensor = torch.tensor(story, dtype=torch.long)
        for i in range(0, len(story_tensor) - seq_len, seq_len):
            tensor_data.append(story_tensor[i:i+seq_len])
    return torch.stack(tensor_data)

train_data = data_to_tensor_bpe(tokenized_train_stories, bpe_tokenizer, seq_len)
train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

def save_vocab(vocab, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(vocab, f)

vocab_path = "vocab.pkl"
# save_vocab(vocab, vocab_path)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5, device='cpu'):
        super(LSTMModel, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=True, batch_first=True)  # Add batch_first=True
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)  # Pass x directly
        x = x.contiguous().view(-1, self.hidden_size * 2)  # Reshape x
        x = self.fc(x)
        return x, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers * 2, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers * 2, batch_size, self.hidden_size))


# model = LSTMModel(len(vocab), embedding_dim, hidden_dim, num_layers, dropout_prob).to(device)
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob).to(device)
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(ignore_index=bpe_tokenizer.token_to_id("<pad>"))
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Add learning rate scheduler
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # Changed to Exponential scheduler with gamma decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)


# Load and preprocess validation set
val_file_path = 'kids_stories_val_new.txt'
# val_data = process_data(val_file_path, tokenizer)
val_data = process_data_bpe(val_file_path, bpe_tokenizer)
val_data = data_to_tensor_bpe(val_data, bpe_tokenizer, seq_len)
val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

# Load and preprocess test set
test_file_path = 'kids_stories_test_new.txt'
# test_data = process_data(test_file_path, tokenizer)
test_data = process_data_bpe(test_file_path, bpe_tokenizer)
test_data = data_to_tensor_bpe(test_data, bpe_tokenizer, seq_len)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            x, target = batch[:-1], batch[1:]

            hidden = model.init_hidden(x.size(0))  # Use x.size(0) to get the actual batch size  
            hidden = tuple([h.detach() for h in hidden])
            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, vocab_size), target.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(train_data):
        batch = batch.to(device)
        x, target = batch[:-1], batch[1:]
        optimizer.zero_grad()
        
        hidden = model.init_hidden(x.size(0))  # Use batch.size(0) to get the actual batch size
        hidden = tuple([h.detach() for h in hidden])
        output, hidden = model(x, hidden)
        # loss = criterion(output.view(-1, len(vocab)), target.view(-1))
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch: {epoch + 1}/{epochs}, Step: {i + 1}, Loss: {total_loss / (i + 1)}')
    
    avg_train_loss = total_loss / len(train_data)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))
    print(f'Epoch: {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Training Perplexity: {train_perplexity}')

    # Validation
    val_loss = evaluate(model, val_data)
    print(f'Epoch: {epoch + 1}/{epochs}, Validation Loss: {val_loss}, Validation Perplexity: {torch.exp(torch.tensor(val_loss))}')
    
    # Call scheduler.step(val_loss) after each epoch
    scheduler.step(val_loss)

    # scheduler.step()

# Save the trained model
torch.save(model, "trained_lstm_model.pth")

# Save the tokenizer
# with open("tokenizer.pkl", "wb") as f:
#     pickle.dump(tokenizer, f)
with open("bpe_tokenizer.pkl", "wb") as f:
    pickle.dump(bpe_tokenizer, f)



# Training loop with validation
for epoch in range(epochs):
    # Training ...
    
    # Validation
    val_loss = evaluate(model, val_data)
    print(f'Epoch: {epoch + 1}/{epochs}, Validation Loss: {val_loss}, Validation Perplexity: {torch.exp(torch.tensor(val_loss))}')

test_loss = evaluate(model, test_data)
print(f'Test Loss: {test_loss}, Test Perplexity: {torch.exp(torch.tensor(test_loss))}')

