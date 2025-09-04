#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A Candidate for the MSc in Digital Scholarship 2024-25, University of Oxford

NB: ChatGPT was used to "vibe-code" much of the contents of this notebook and debug manually created sections.

Notes on Application in the HathiAnalytics Research Data Capsule: 

- The resources within Hathi were still running on Python 3.6 so this script was written accordingly.
- All of the packages were loaded with version that were compatible with 3.6, noted below.

"""

# === Importing Packages ===
import os, random, time, json, zipfile
import numpy as np # 1.19.5
import pandas as pd # version 1.1.5
import torch  # version 1.10.2
from transformers import BigBirdTokenizer, BigBirdModel # version 4.5.1
import tqdm # version 4.64.1
import logging # version 0.5.1.2
import sentencepiece # version 0.2.0
import re
import ast
import sys

# === LOADING MODELS ===

from transformers import BigBirdTokenizer, BigBirdModel

model_path = "/home/dcuser/NovelVectTest/bigbird-roberta-base-export"

tokenizer = BigBirdTokenizer.from_pretrained(model_path)
model = BigBirdModel.from_pretrained(model_path)

# === GLOBAL VARIABLES ===
DIRECTORY_PATH = "/media/secure_volume/2000NovleVect/2000NovelsContent"
WINDOW_LENGTH = 2048
STORAGE_PATH = "/media/secure_volume/2000NovelVect/2000NovelsData"
TOKEN_BUDGET = 0 # Upper limit of what model can batch at once

# Note that batching was attempted, but this overloaded the memory. 
# The batching limit was subsequently reduced to zero in order to effectively stop batching.

# === LOADING FILES ===

# Get the file names
def get_filenames(directory_path):
    files = []
    for f in os.listdir(directory_path):
        if not f.endswith(".txt"):
            print(f"Non-.txt file detected: {f}")
        if f.endswith(".txt"):
            files.append(f)
    return files

# Open files

def open_file(filepath):
        with open(filepath,'r', encoding = "utf-8") as f:
            if f: 
                return f.read()
            else: return print(f"Unable to open file: {f}")
    
# === CHOOSING TEST SET ===

# Get test set of 100

def select_100(file_list):
    random.seed(22)
    return random.sample(file_list, k=100)

# This set was used to test the vectorization prodcedure in the Hathi Research Data Capsule Environment
    

# === LIST PARSING HELPER ===

def safe_count(x):
    if isinstance(x, (list, tuple)) and x:
        return len(x)
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple)) and parsed:
                return len(parsed)
        except:
            return 0
    return 0

# === CHUNKING ===

def get_chunks(file, window_length = WINDOW_LENGTH): 
    totalBookTokens = 0
    chunk_lengths = []
    split_paragraphs =[]
    chunks = []
    
    text = open_file(file)
    if not text.strip():
        raise ValueError(f"[Empty file] {file} is empty or unreadable.")

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        raise ValueError(f"[No paragraphs] {file} could not be split into paragraphs.")
    
   
    # Tokenize each paragraph
    tokenized_paragraphs = [tokenizer.tokenize(para) for para in paragraphs ]
    if not tokenized_paragraphs:
        raise ValueError(f"[No tokens] {file} has no tokenized content.")
    
    # Track the index of the paragraphs 
    i = 0 
    while i < len(tokenized_paragraphs):
        tokens = tokenized_paragraphs[i]
        totalBookTokens += len(tokens)
    
        
        # Overlap logic
        if i > 1: 
            overlap_len = int(len(tokenized_paragraphs[i - 1]) * 0.10)
            overlap = tokenized_paragraphs[i - 1][-overlap_len:] if overlap_len > 0 else []
            extended_tokens = overlap + tokens
            
        else:
            extended_tokens = tokens
            
        original_len = len(tokens)
       
        # If chunck excedes the window length split it down the middle
        if len(extended_tokens) > window_length:
            print(f"Note: Paragraph split on paragraph index {i}.")
            split_paragraphs.append(i) 
            split = len(extended_tokens) // 2 # Better than just truncating because you might be stuck with a stubby incoherent segment
            tokens1 = tokens[:split]
            tokens2 = tokens[split:]
            
            chunk_lengths.extend([original_len // 2, original_len - original_len // 2])
            
            # Extend second partition of tokens 
            overlap2_len = int(len(tokens1) * 0.10)
            extended_tokens2 = tokens2 + tokens1[:overlap2_len] if overlap2_len > 0 else tokens2
            
            # Add to chunks list 
            chunks.extend([tokens1, extended_tokens2])
            
        else: 
            chunks.append(extended_tokens)
            chunk_lengths.append(original_len)
        
        i += 1
            
    return chunks, chunk_lengths, totalBookTokens, split_paragraphs

# === GETTING CHUNK WEIGHTS ===

def get_chunk_weights(chunks, chunk_lengths, totalBookTokens): 
    chunk_weights = {}
    
    i = 0 
    for length in chunk_lengths:
        weight = length/totalBookTokens
        chunk_weights[i] = weight
        i += 1
    return chunk_weights
    

# === EMBEDDING LOGIC ===

# Embed individual chunks
def embed_chunk(chunk, max_length=WINDOW_LENGTH):
    text_chunk = " ".join(chunk)
    inputs = tokenizer(text_chunk, return_tensors = 'pt', truncation = True, 
                      padding = True, max_length = max_length)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Embed token batch
def embed_batch(batch_texts, batch_weights):
    try:
        inputs = tokenizer(batch_texts, return_tensors = 'pt',
                           padding=True, truncation=True, max_length=WINDOW_LENGTH)
        outputs = model(**inputs)
        embeds = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return [emb * w for emb, w in zip(embeds, batch_weights)]
    except RuntimeError as e:
        print(f"[Memory Error] Reducing batch size: {e}")
        fallback =[]
        for text, w in zip(batch_texts, batch_weights):
            try:
                inputs = tokenizer(text, return_tensors='pt', padding=True,
                                   truncation=True, max_length=WINDOW_LENGTH)
                output = model(**inputs)
                emb = output.last_hidden_state.mean(dim=1).detach().numpy()
                fallback.append(emb * w)
            except Exception as e2:
                print(f"[Skip chunk] {e2}")
        return fallback
# Weighted average of embeddings to get entire book
def embed_book(chunks, weights): 
    
    current_batch = []
    current_weights = []
    current_token_count = 0 
    weighted_embeddings = []
    
    if not TOKEN_BUDGET or TOKEN_BUDGET <= 0: 
        for i, chunk in enumerate(chunks):
            text = " ".join(chunk)
            weighted_embeddings.extend(embed_batch([text], [weights[i]]))
        if len(weighted_embeddings) == 0:
            raise ValueError("No embeddings generated.")
        book_embedding = sum(weighted_embeddings) / len(weighted_embeddings)
        return book_embedding

# === TRACKING VECTORIZATION TIME ===

def track_time(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    time_elapsed = end_time - start_time
    return result, time_elapsed

# === CREATE BOOK VECTOR ===

def create_book_vectors(directory_path, storage_path): 
    
    file_list = get_filenames(directory_path)
    
    rows_buffer = []
    
    for i, file in enumerate(tqdm.tqdm(file_list, desc="Vectorizing Books")): 
        try:
            filepath = os.path.join(directory_path, file)
            chunks, chunk_lengths, totalBookTokens, split_paragraphs = get_chunks(filepath)
            weights = get_chunk_weights(chunks, chunk_lengths, totalBookTokens)
            book_vector, time_elapsed = track_time(embed_book, chunks, weights)
        except Exception as e: 
            print(f"[Failure] Processing failed for {file}: {e}.")
            continue
            
        row = {
            "ExtendedTokenLength": sum([len(chunk) for chunk in chunks]) if chunks else print("[Error] chunks empty!"),
            "ChunkLengths": chunk_lengths,
            "TotalBookTokens": totalBookTokens,
            "SplitParagraphs": split_paragraphs,
            "BookVector": book_vector.tolist(),
            "TimeElapsed": time_elapsed,
            "FileName": file}
        if not row: 
            print(f"Failure to add row to dataframe. Affected file: {file}")
            
        rows_buffer.append(row)
        
        file_out = os.path.join(storage_path, "BookVectorDataREAL.csv")
        if (i +1)% 5 == 0 or (i+1) == len(file_list):
            if os.path.exists(file_out):
                existing_df = pd.read_csv(file_out)
                vector_data = pd.concat([existing_df, pd.DataFrame(rows_buffer)], ignore_index=True)
            else:
                print("Initialization of vector_data dataframe. Should only occur once per run.")
                vector_data = pd.DataFrame(rows_buffer)
                
            vector_data.to_csv(file_out, index=False)
            print(f"[Checkpoint] saved {i+1} rows to {file_out}")
            rows_buffer =[]
        
    print("\n=== Summary Report ===")
    print(f"Total Novels Processed: {len(vector_data.index)}")
    print(f"Average Vectorization Time: {round(np.mean(vector_data['TimeElapsed']))} seconds") 
    num_splits = vector_data['SplitParagraphs'].apply(safe_count).sum()
    print(f"Number of Paragraph Splits: {num_splits}")
    
    return vector_data 

# === RUNNING ===

vectorData_df, total_time = track_time(create_book_vectors, DIRECTORY_PATH, STORAGE_PATH)

print(f"[Done] Vectorization round completed in {round(total_time,2)} seconds.")