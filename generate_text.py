import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

from lstm_language_model import LSTMModel

def load_vocab(file_path):
    with open(file_path, "rb") as f:
        vocab = pickle.load(f)
    stoi = {token: vocab[token] for token in vocab.get_itos()}
    itos = {idx: token for idx, token in enumerate(vocab.get_itos())}
    return stoi, itos


def tokenize_text(text, stoi):
    tokens = [stoi[token] for token in text.split() if token in stoi]
    return tokens

def tokens_to_text(tokens, itos):
    text = ' '.join(itos[token] for token in tokens)
    return text

def generate_text(model, stoi, itos, initial_text, temperature, text_length):
    model.eval()
    tokens = tokenize_text(initial_text, stoi)
    hidden = model.init_hidden(1)

    with torch.no_grad():
        for _ in range(text_length):
            input_tensor = torch.tensor([tokens[-1]]).unsqueeze(0).to(device)
            output, hidden = model(input_tensor, hidden)

            probabilities = torch.softmax(output / temperature, dim=-1).squeeze()
            next_token = torch.multinomial(probabilities, 1).item()

            tokens.append(next_token)

    generated_text = tokens_to_text(tokens, itos)
    return generated_text

if __name__ == "__main__":
    model_path = "trained_lstm_model.pth"
    vocab_path = "vocab.pkl"  # Add this line; make sure it points to the correct file
    model = torch.load(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    stoi, itos = load_vocab(vocab_path)  # Add this line after loading the model

    initial_text = input("Enter your initial text: ")  # Get input from the user
    temperature = 1.0
    text_length = 200  # Adjust this value to control the length of the generated story

    generated_text = generate_text(model, stoi, itos, initial_text, temperature, text_length)  # Update this line
    print(generated_text)

