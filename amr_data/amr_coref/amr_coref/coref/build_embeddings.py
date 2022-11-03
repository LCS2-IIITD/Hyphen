import re
import difflib
from   tqdm import tqdm
from   multiprocessing import Pool
from   .vocab_embeddings import normalize_token


def build_embeddings(embed_in_dict, gdata_dict):
    # Get all tokens from the serialized graph and normalize them
    # normalize_token is called the Vocab class before getting the emebdding index.
    token_set = set()
    for gdata in gdata_dict.values():
        tokens  = gdata['sgraph'].split()
        token_set.update([normalize_token(t) for t in tokens])
    token_set.discard(None) # normalize can returns None for empty tokens
    # Put everything in sets and print some stats
    embed_in_set = set(embed_in_dict)
    missing_set = token_set - embed_in_set
    common_set  = token_set & embed_in_set
    print('There are {:,} total tokens in the emebdding set'.format(len(embed_in_set)))
    print('There are {:,} total tokens in common'.format(len(common_set)))
    print('There are {:,} missing initially'.format(len(missing_set)))
    # Add some known translation for edge tokens
    proxy_dict0 = {}
    proxy_dict0, missing_set = edge_match(missing_set, embed_in_set)
    print('There are {:,} missing after edge matching'.format(len(missing_set)))
    # Now do some more fuzzy matching to try to map unknown simple attribs to tokens in the embedding file
    proxy_dict1, missing_set = simple_match(missing_set, embed_in_set)
    print('There are {:,} missing after simple matching'.format(len(missing_set)))
    proxy_dict2, missing_set = fuzzy_match(missing_set, embed_in_set)
    print('There are {:,} missing after difflib matching'.format(len(missing_set)))
    # Combine the proxy dictionaries
    proxy_dict = {**proxy_dict0, **proxy_dict1, **proxy_dict2}
    final_embed_set = common_set
    # Add in all the GloVe tokens that are needed as vectors for proxy token
    final_embed_set.update(proxy_dict.values())
    # Sanity check
    for token in sorted(token_set):
        if token not in final_embed_set and token not in missing_set:
            assert token in proxy_dict
    print('There are {:,} final embedding tokens'.format(len(final_embed_set)))
    # Filter the original embedding dict for words in the new vocabalulary
    embed_out_dict = {k:v for k, v in embed_in_dict.items() if k in final_embed_set}
    # Copy (duplicate) existing embedding vectors to proxy names the proxy_dict
    for proxy_token, glove_token in proxy_dict.items():
        embed_out_dict[proxy_token] = embed_out_dict[glove_token]
    print('There are {:,} tokens after appling proxies'.format(len(embed_out_dict)))
    # For debug
    if 1:
        pdfn = '/tmp/proxy_dict.txt'
        print('Debug proxy dict written to', pdfn)
        with open(pdfn, 'w') as f:
            for k, v in sorted(proxy_dict.items()):
                f.write('%-20s : %s\n' % (k, v))
    return embed_out_dict


# Translate opX and argX to op and arg, which are in the embeddings
re_op  = re.compile(r'^op\d+$')
re_arg = re.compile(r'^arg\d+$')
def edge_match(missing_set, embed_set):
    missing_set = missing_set.copy()
    proxy_dict = {}
    for token in sorted(missing_set):
        if re_op.search(token):
            proxy_dict[token] = 'op'
            missing_set.remove(token)
        elif re_arg.search(token):
            proxy_dict[token] = 'arg'
            missing_set.remove(token)
    return proxy_dict, missing_set


# Do some simple matching
def simple_match(missing_set, embed_set):
    missing_set = missing_set.copy()
    proxy_dict = {}
    for token in sorted(missing_set):
        # check for integers and replace with something known to be in the set
        # The original embeddings have 0,1,.. so the only integers missing are
        # larger values.
        if token.isnumeric():
            proxy_dict[token] = '1000'
            missing_set.remove(token)
            assert '10' in embed_set
        # Replace words with dashes with a partial word
        elif '-' in token:
            for test in token.split('-'):
                if test in embed_set:
                    proxy_dict[token] = test
                    missing_set.remove(token)
                    break
        # Replace words with underscores with a partial word
        elif '_' in token:
            for test in token.split('_'):
                if test in embed_set:
                    proxy_dict[token] = test
                    missing_set.remove(token)
                    break
    return proxy_dict, missing_set


# Do more expensive partial string matching
def fuzzy_match(missing_set, embed_set):
    missing_set = missing_set.copy()
    proxy_dict = {}
    if not missing_set:
        return proxy_dict, missing_set
    # Do multiprocessing matching
    global g_embed_set
    g_embed_set = embed_set
    missing_list = sorted(missing_set)
    with Pool() as pool:
        for i, proxy_token in enumerate(pool.imap(difflib_worker, missing_list, chunksize=1)):
            if proxy_token is not None:
                assert proxy_token in embed_set
                token = missing_list[i]
                proxy_dict[token] = proxy_token
                missing_set.remove(token)
    return proxy_dict, missing_set


# Worker function to run difflib matching in multiprocessing pool
g_embed_set = None
def difflib_worker(token):
    global g_embed_set
    matches = difflib.get_close_matches(token, g_embed_set, n=1, cutoff=0.6)
    if len(matches) > 0:
        return matches[0]
    else:
        return None
