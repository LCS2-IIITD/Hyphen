import re
from  .word_vectors import load_embeddings_with_specials


class Vocab(object):
    def __init__(self, tokens):
        self.idx2tok  = tokens
        self.tok2idx  = {t:i for i, t in enumerate(self.idx2tok)}
        self.none_idx = self.tok2idx['<none>']   # Used for indexing outside of boundaries
        self.unk_idx  = self.tok2idx['<unk>']    # For any token not in the vocab

    def get_token(self, index):
        return self.idx2tok[index]

    def get_index(self, token):
        token = normalize_token(token)
        if token is None:
            return self.none_idx
        return self.tok2idx.get(token, self.unk_idx)

    def get_embedding_tokens(self):
        return self.idx2tok

    def __len__(self):
        return len(self.idx2tok)


# Normalize graph tokens to (somewhat) match glove emebedding tokens
re_strip_enum = re.compile(r'-\d+$')    # remove sense tag numbers
re_edge_id    = re.compile(r'^:')       # remove leading colon on edges
def normalize_token(token):
    if token is None:
        return None
    token = token.lower()
    if 'http' in token or 'www.' in token or '.com' in token:
        return None
    token = token.replace(' ', '_')
    token = token.replace('"', '')
    token = token.replace("'s", '')
    token = token.replace("'t", '')
    token = token.replace(",", '')
    token = re_strip_enum.sub('', token)
    token = re_edge_id.sub('', token)
    token = token if token else None
    return token


# Vocab comes from the tokens in the embeddings so load both together
def load_vocab_embeddings(embedding_fpath):
    tokens, embed_mat = load_embeddings_with_specials(embedding_fpath)
    vocab = Vocab(tokens)
    return vocab, embed_mat


# method to load a list of words
def load_word_set(fpath):
    if fpath is None:
        return None
    with open(fpath) as f:
        lines = [l.strip() for l in f.readlines()]
    words = [l for l in lines if l]
    return  set(words)
