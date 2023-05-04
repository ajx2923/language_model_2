class Vocab:
    def __init__(self, tokens):
        self.tokens = tokens
        self.stoi = {token: i for i, token in enumerate(tokens)}
        self.itos = {i: token for i, token in enumerate(tokens)}

    def __getitem__(self, key):
        return self.stoi[key]

    def __contains__(self, key):
        return key in self.stoi

    def get_itos(self):
        return self.itos