from datasets import load_dataset #pip install datasets
from tqdm import tqdm 
import numpy as np
import tiktoken
import os

#------------------------------------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

#Init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] #end of text tokens
def tokenizer(doc):
    #Tokenizes a single document and returns a numpy array of unit16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token directory too large"
    tokens_np_unit16 = tokens_np.astype(np.unit16)
    return tokens_np_unit16

def write_datafile(filename, tokens_np):
    #Writes a numpy array of unit16 tokens to a binary file
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())

# tokenize all documents and write output shards, each of shard_size tokens
nprocs = max(1, os.cpu_count()//2)
with np.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size), dtype=np.unit16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        #is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            #Simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_size}")
                progress_bar.update(len(tokens))
        else:
            #write current shard to disk and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufinweb_{split}_{shard_size}")
            #Split the document into whatever fits in this shard;
            reminder = shard_size - token_count
            progress_bar.update(reminder)
            all_tokens_np[token_count:token_count+reminder] = tokens[:reminder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # Populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-reminder] = tokens[reminder:]
            token_count = len(tokens)-reminder

    #Write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufinweb_{split}_{shard_index}")
        write_datafile(filename, all_tokens_np[:token_count])


