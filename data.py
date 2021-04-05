""" Tools for loading datasets as Classification/SequenceLabelling Examples. """
import os
import logging
from collections import namedtuple

from tqdm import tqdm
from transformers import BasicTokenizer
import pandas as pd
from utils.data import retokenize
import text_cleaning_transforerms as tc
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/'
ClassificationExample = namedtuple(
    'ClassificationExample', ['id', 'tokens_a', 'tokens_b', 'label'])
SequenceLabellingExample = namedtuple(
    'SequenceLabellingExample', ['id', 'token_sequence', 'label_sequence'])


def load_classification_dataset(step, do_lower_case):
    """ Loads classification exampels from a dataset. """
    assert step in ['train', 'test']
    binary = True   
    path_data = '~/Github/Data/Patient/NIRADS/PET_CT_NIRADS.xlsx'
    data_CT = pd.read_excel(path_data)
    data,_, y, _  = tc.text_cleaning(data_CT, None, data_target='section')
    if binary:
        y[y<2]=0
        y[y>0]=1

    train_text, test_text, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1)

    if step =='train':
        data_to_use = train_text.copy()
        y_to_use = y_train.copy()
    else:
        data_to_use = test_text.copy()
        y_to_use = y_test.copy()

    basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    examples = []

    for i, tokens in tqdm(enumerate(data_to_use)):
        label = y_to_use[i]
        examples.append(
            ClassificationExample(
                id=i,
                tokens_a=basic_tokenizer.tokenize(tokens),
                tokens_b=None,
                label=label,
            )
        )
    logging.info('Number of `%s` examples: %d', step, len(examples))
    
    return examples

def load_sequence_labelling_dataset(step, do_lower_case):
    """ Loads sequence labelling examples from a dataset. """
    assert step in ['train', 'test']
    path = os.path.join(DATA_PATH, 'sequence_labelling', f'{step}.txt')
    i = 0
    examples = []
    with open(path, 'r', encoding='utf-8') as data_file:
        lines = data_file.readlines()
        token_sequence = []
        label_sequence = []
        for line in tqdm(lines, desc=f'reading `{os.path.basename(path)}`...'):
            # example:
            #          My O
            #          name O
            #          is O
            #          Hicham B-PER
            #          . O
            splitline = line.strip().split()
            if splitline:
                token, label = splitline
                token_sequence.append(token)
                label_sequence.append(label)
            else:
                examples.append(
                    SequenceLabellingExample(
                        id=i,
                        token_sequence=token_sequence,
                        label_sequence=label_sequence,
                    )
                )
                i += 1
                token_sequence = []
                label_sequence = []

    # Don't forget to add the last example
    if token_sequence:
        examples.append(
            SequenceLabellingExample(
                id=i,
                token_sequence=token_sequence,
                label_sequence=label_sequence,
            )
        )

    retokenize(
        examples,
        tokenization_function=BasicTokenizer(do_lower_case=do_lower_case).tokenize)
    logging.info('Number of `%s` examples: %d', step, len(examples))
    return examples



if __name__ == '__main__':
    a = load_classification_dataset('test', True)
    print(a[0].label)

