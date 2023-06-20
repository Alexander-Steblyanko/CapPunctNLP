import pandas
import torch
import transformers
import numpy as np
import pandas as pd
import os

from settings import *

tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(bert_model_name)


def span_init():
    return pd.Series(index=['x', 'y', 'x_mask', 'y_mask'], dtype=object,
                     data=[[], [], [], []])


def line_process(line):
    # Create text span for line
    t_span = span_init()
    # Create dataframe to deposit spans into
    df = pd.DataFrame(columns=['x', 'y', 'x_mask', 'y_mask'], dtype=object)

    # Initial values
    t_span['x'].append(spec_tokens['START_SEQ'])
    t_span['y'].append(0)
    t_span['y_mask'].append(0)

    # Remove whitespace
    line = line.strip()

    # Transform text for easier processing
    line = line.replace(" ...", ".") \
        .replace("!", ".") \
        .replace(" -- ", "- ") \
        .replace(":", "- ") \
        .replace(";", ",") \
        .replace("?!", "?") \
        .replace(" '", " ").replace("' ", " ") \
        .replace("[", "").replace("]", "") \
        .replace("\"", "")  # May cause uppercasing issues

    # Split line into words
    words = line.split()

    # Process each word
    for w in words:
        p = 0

        # Determine case
        if w[0].isupper():
            p += punc_dict['_Up']

        # Determine punct
        end = w[-1]
        if end == '.':
            w = w[:-1]
            p += punc_dict['.']
        elif end == ',':
            w = w[:-1]
            p += punc_dict[',']
        elif end == '?':
            w = w[:-1]
            p += punc_dict['?']
        elif end == '-':
            w = w[:-1]
            p += punc_dict['-']

        # Lowercase word
        w = w.lower()

        # Tokenize word
        tokens = tokenizer.tokenize(text=w)

        # Sort into batches:
        # If text span is almost-full, fill it w/ padding & start a new one
        if len(tokens) + len(t_span['x']) >= sequence_len:
            # Add padding
            if len(t_span['x']) < sequence_len:
                [t_span['x'].append(spec_tokens['PAD']) for _ in range(sequence_len - len(t_span['x']))]
                [t_span['y'].append(0) for _ in range(sequence_len - len(t_span['y']))]
                [t_span['y_mask'].append(0) for _ in range(sequence_len - len(t_span['y_mask']))]

            # Finalize text span
            t_span['x_mask'] = [1 if token != spec_tokens['PAD'] else 0 for token in t_span['x']]
            df = pandas.concat([df, t_span.to_frame().T], ignore_index=True)

            # Start up a new batch
            t_span = span_init()

        # Add word tokens to span
        if len(tokens) > 0:
            # If several tokens, all non-final tokens are punct-less
            for i in range(len(tokens) - 1):
                t_span['x'].append(tokenizer.convert_tokens_to_ids(tokens[i]))
                t_span['y'].append(0)
                t_span['y_mask'].append(0)
            # Add in the last token (final punct class at the end)
            t_span['x'].append(tokenizer.convert_tokens_to_ids(tokens[-1]))
            t_span['y'].append(p)
            t_span['y_mask'].append(1)
        else:
            t_span['x'].append(spec_tokens['UNKNOWN'])
            t_span['y'].append(p)
            t_span['y_mask'].append(1)

    # Once out of words, finalize the last span of line
    # Add end of sequence token
    t_span['x'].append(spec_tokens['END_SEQ'])
    t_span['y'].append(0)
    t_span['y_mask'].append(0)

    # Add padding
    if len(t_span['x']) < sequence_len:
        [t_span['x'].append(spec_tokens['PAD']) for _ in range(sequence_len - len(t_span['x']))]
        [t_span['y'].append(0) for _ in range(sequence_len - len(t_span['y']))]
        [t_span['y_mask'].append(0) for _ in range(sequence_len - len(t_span['y_mask']))]

    # Finalize batch
    t_span['x_mask'] = [1 if token != spec_tokens['PAD'] else 0 for token in t_span['x']]
    df = pandas.concat([df, t_span.to_frame().T], ignore_index=True)
    return df


def reinterpret_tokens(x, y, x_mask, y_mask):
    idx = 0
    sentence = ""
    word = ""

    while idx < len(x) and x_mask[idx]:
        #print(x[idx], y[idx])
        if x[idx] not in spec_tokens.values() or x[idx] == spec_tokens['UNKNOWN']:
            word += tokenizer.decode(x[idx], clean_up_tokenization_spaces=True)
            if y_mask[idx]:
                if y[idx] >= punc_dict['_Up']:
                    word = word[0].upper() + word[1:]
                    y[idx] -= punc_dict['_Up']
                sentence += word
                word = ""

                if y[idx] == punc_dict[',']:
                    sentence += ","
                elif y[idx] == punc_dict['.']:
                    sentence += "."
                elif y[idx] == punc_dict['?']:
                    sentence += "?"
                elif y[idx] == punc_dict['-']:
                    sentence += " -"
                sentence += " "
        idx += 1
    return sentence


def file_process(file):
    # DataFrame of text spans
    df = pd.DataFrame(columns=['x', 'y', 'x_mask', 'y_mask'], dtype=object)

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            l_df = line_process(line)
            df = pandas.concat([df, l_df], ignore_index=True)

            if df.shape[0] % 5000 == 0:
                print(df)

    # Showcase df
    print(df)

    # Save as hdf5
    df.to_hdf(dataset_path, "df", mode='w')

    return df


def file_process_fused(file):
    # DataFrame of text spans
    df = pd.DataFrame(columns=['x', 'y', 'x_mask', 'y_mask'], dtype=object)
    # Create text span for line
    t_span = span_init()

    def len_check(tokens, t_span, df):
        # If text span is almost-full, fill it w/ padding & start a new one
        if len(tokens) + len(t_span['x']) > sequence_len:
            # Add padding
            if len(t_span['x']) < sequence_len:
                [t_span['x'].append(spec_tokens['PAD']) for _ in range(sequence_len - len(t_span['x']))]
                [t_span['y'].append(0) for _ in range(sequence_len - len(t_span['y']))]
                [t_span['y_mask'].append(0) for _ in range(sequence_len - len(t_span['y_mask']))]

            # Finalize text span
            t_span['x_mask'] = [1 if token != spec_tokens['PAD'] else 0 for token in t_span['x']]
            df = pandas.concat([df, t_span.to_frame().T], ignore_index=True)

            if df.shape[0] % 1000 == 0:
                print(df)
            # Start up a new batch
            t_span = span_init()
        return t_span, df

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            # Initial values
            t_span, df = len_check([spec_tokens['START_SEQ']], t_span, df)
            t_span['x'].append(spec_tokens['START_SEQ'])
            t_span['y'].append(0)
            t_span['y_mask'].append(0)

            # Remove whitespace
            line = line.strip()

            # Transform text for easier processing
            line = line.replace(" ...", ".") \
                .replace("!", ".") \
                .replace(" -- ", "- ") \
                .replace(":", "- ") \
                .replace(";", ",") \
                .replace("?!", "?") \
                .replace("'", "") \
                .replace("[", "").replace("]", "") \
                .replace("\"", "")  # May cause uppercasing issues

            # Split line into words
            words = line.split()

            # Process each word
            for w in words:
                p = 0

                # Determine case
                if w[0].isupper():
                    p += punc_dict['_Up']

                # Determine punct
                end = w[-1]
                if end == '.':
                    w = w[:-1]
                    p += punc_dict['.']
                elif end == ',':
                    w = w[:-1]
                    p += punc_dict[',']
                elif end == '?':
                    w = w[:-1]
                    p += punc_dict['?']
                elif end == '-':
                    w = w[:-1]
                    p += punc_dict['-']

                # Lowercase word
                w = w.lower()

                # Tokenize word
                tokens = tokenizer.tokenize(text=w)

                # Add word tokens to span
                if len(tokens) > 0:
                    # Batch check
                    t_span, df = len_check(tokens, t_span, df)
                    # If several tokens, all non-final tokens are punct-less
                    for i in range(len(tokens) - 1):
                        t_span['x'].append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        t_span['y'].append(0)
                        t_span['y_mask'].append(0)
                    # Add in the last token (final punct class at the end)
                    t_span['x'].append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    t_span['y'].append(p)
                    t_span['y_mask'].append(1)
                else:
                    # Batch check
                    t_span, df = len_check([spec_tokens['UNKNOWN']], t_span, df)
                    # Add unknown token
                    t_span['x'].append(spec_tokens['UNKNOWN'])
                    t_span['y'].append(p)
                    t_span['y_mask'].append(1)

            # Once out of words, finalize the last span of line
            # Batch check
            t_span, df = len_check([spec_tokens['END_SEQ']], t_span, df)
            # Add end of sequence token
            t_span['x'].append(spec_tokens['END_SEQ'])
            t_span['y'].append(0)
            t_span['y_mask'].append(0)

            # if df.shape[0] % 5000 == 0:
            #    print(df)

    # Showcase df
    print(df)

    # Save as hdf5
    df.to_hdf(dataset_path, "df", mode='w')


if __name__ == '__main__':
    file = 'data/IWSLT12.TALK.train.en'
    file_process(file)
