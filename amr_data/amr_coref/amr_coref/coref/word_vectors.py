import numpy
from   collections import OrderedDict


# Load the raw word2vect (aka GloVe) style embedding file
def load_embeddings(fpath, tokens_only=False, vocab_set=None):
    # Load every dataline
    with open(fpath) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [l         for l in lines if l]
    # Process the header (if it has one)
    wcount, vec_dim = 0, 0
    if len(lines[0].split()) == 2:
        header  = lines.pop(0).split()
        wcount  = int(header[0])
        vec_dim = int(header[1])
    # Convert lines to embedding vectors
    embed_dict = OrderedDict()
    for line in lines:
        sp = line.split()
        token = sp[0].strip()
        if token and (vocab_set is None or token in vocab_set):
            if tokens_only:
                embed_dict[token] = None
            else:
                embed_dict[token] = [float(x) for x in sp[1:]]
    # Get vec_dim if we don't already have it
    if vec_dim == 0 and not tokens_only:
        item = next(iter(embed_dict.values()))
        vec_dim = len(item)
    # Return a dictionary of the vocabulary with vectors
    return embed_dict, vec_dim


# Load the embedding file and add special tokens for vocab
def load_embeddings_with_specials(fpath):
    embed_dict, vec_dim = load_embeddings(fpath)
    # Convert lines to embedding vectors
    # <none> ==> None, out of bounds, or pad.  <unk> is unknown (average vector)
    zero_vec = [0.0]*vec_dim
    tokens   = ['<none>', '<unk>']  + list(embed_dict.keys())
    vectors  = [zero_vec, zero_vec] + list(embed_dict.values())
    # Convert to numpy and find the average for <unk>
    embed_mat    = numpy.array(vectors, dtype='float32')
    embed_mat[1] = numpy.mean(embed_mat[2:,:], axis=0)
    # Error check
    assert len(tokens) == embed_mat.shape[0]
    assert vec_dim     == embed_mat.shape[1]
    # Return data
    return tokens, embed_mat


# Save an extracted portionn of an embedding dict
def save_embeddings(embed_dict, fpath):
    with open(fpath, 'w') as f:
        for token, vector in sorted(embed_dict.items()):
            vec_str = ' '.join(['%8.5f' % v for v in vector])
            f.write('%s %s\n' % (token, vec_str))
