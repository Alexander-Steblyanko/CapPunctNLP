import torch
import pandas as pd

from model import *
from pdata import *

# Load in model
ans_model = PunctuationModel(bert_model_name)
ans_model.load_state_dict(torch.load(model_load_path))
ans_model.eval()


def answer(text):
    inbetween = pd.Series(index=['x', 'y_mask', 'y_pred'], dtype=object,
                          data = [[], [], []])

    # Line processing sends back a dataframe, may contain several batches
    df_tokens = line_process(text)
    for r in range(len(df_tokens)):
        x = df_tokens.loc[r]['x']
        x_mask = df_tokens.loc[r]['x_mask']
        y_mask = df_tokens.loc[r]['y_mask']
        # Record (string repr of) generated tokens
        [inbetween['x'].append(tokenizer.decode(x[i])) for i in range(len(x))
             if x_mask[i] == 1]
        [inbetween['y_mask'].append(y_mask[i]) for i in range(len(y_mask))
             if x_mask[i] == 1]

        with torch.no_grad():
            corr_text = ""
            # For each batch... (we ignore the generated y - it's bunk/empty)
            # Get predicted punctuation
            y_pred = ans_model(torch.tensor(x).view(1, -1),
                               torch.tensor(x_mask).view(1, -1))
            y_pred = y_pred.view(-1, y_pred.shape[2])
            y_pred = torch.argmax(y_pred, dim=1).view(-1)
            # Record (string repr of) predicted punctuation
            [inbetween['y_pred'].append(list(punc_dict.values()).index(y_pred[i]))
             for i in range(len(y_pred)) if x_mask[i] == 1]
            # Use predicted punctuation to correct text
            corr_text += reinterpret_tokens(x, y_pred, x_mask, y_mask)

    return corr_text, inbetween


if __name__ == '__main__':
    #ct, ib = answer("So this is a mantis shrimp. There are the eyes up here, and there's that raptorial appendage, and there's the heel.")
    ct, ib = answer("We made the ocean unhappy; we made people very unhappy, and we made them unhealthy.")
    print(ib['x'])
    print(ib['y_mask'])
    print(ib['y_pred'])
    print(ct)
