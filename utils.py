import os

import numpy as np
import tensorflow as tf
import pandas as pd

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def read_data(file_path):
    """
    Read data into dataframe.
    Columns include subject IDs and ICD9 codes.
    """
    data_df = pd.read_csv(file_path)
    return data_df

def read_vocab(file_path):
    """
    Read all ICD9 codes, which make up the vocabulary.
    Columns include ICD9 codes and long and short descriptions of each code.
    """
    vocab_df = pd.read_excel(file_path)
    return vocab_df

def build_vocab(words, vocab_size, visual_fld):
    """
    Build vocabulary of ICD9 codes and write it to visualization/vocab.tsv
    """
    safe_mkdir(visual_fld)
    file = open(os.path.join(visual_fld, 'vocab.tsv'), 'w')
    
    dictionary = dict()
    index = 0
    
    for word in words:
        dictionary[word] = index
        index += 1
        file.write(word + '\n')
    
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    file.close()
    return dictionary, index_dictionary

def write_vocab_descs(descs, visual_fld):
    """
    Write vocabulary descriptions of ICD9 codes to visualization/vocab_descs.tsv
    """
    safe_mkdir(visual_fld)
    file = open(os.path.join(visual_fld, 'vocab_descs.tsv'), 'w')
    
    for desc in descs:
        file.write(desc + '\n')
    
    file.close()

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(data_df):
    """ Form training pairs according to the skip-gram model. """
    # Form groups based on subject
    grouped = data_df.groupby('SUBJECT_ID')

    # Use indexed ICD9 codes within each subject to form training pairs
    for _, group in grouped:
        index_words = group['INDICES'].tolist()

        # For each code, make a training pair with every other code of this
        # subject
        for i in range(len(index_words)):
            for j in range(len(index_words)):
                if (i != j):
                    yield index_words[i], index_words[j]

def batch_gen(vocab_size, batch_size, visual_fld):
    # Local paths to data files
    data_dest = 'data/DIAGNOSES_ICD.csv'
    vocab_dest = 'data/CMS32_DESC_LONG_SHORT_DX.xlsx'

    # Read in data containing subject IDs and ICD9 codes
    data_df = read_data(data_dest)

    # Read in ICD9 codes that make up the vocabulary
    vocab_df = read_vocab(vocab_dest) # also contains string descriptions
    codes = vocab_df['DIAGNOSIS CODE']
    descs = vocab_df['LONG DESCRIPTION']

    # Build vocab dictionary to map codes to indices
    dictionary, _ = build_vocab(codes, vocab_size, visual_fld)

    # Write vocabulary descriptions to file
    write_vocab_descs(descs, visual_fld)

    # Create new column in data containing indices into vocabulary
    data_df['INDICES'] = convert_words_to_index(data_df['ICD9_CODE'], dictionary)

    single_gen = generate_sample(data_df)
    
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(single_gen)
        yield center_batch, target_batch
