import tiktoken

# the in-/out-/request price in USD per M-token
# for ours
MODEL_PRICE = {'gpt-4o-mini':   [0.15, 0.6, 0],
                'gpt-4-turbo':   [10, 30, 0],
                'gpt-4o':        [5, 15, 0],
                'gemini-1.5-flash-002':  [0.075, 0.3, 0],
                'gemini-1.5-pro-002':    [3.5, 10.5, 0],
                'gemini-1.0-pro':    [0.5, 1.5, 0],
                'Phi-3-mini-4k-instruct':     [0.13, 0.52, 0],
                'Phi-3.5-mini-instruct':   [0.13, 0.52, 0],
                'Phi-3-small-8k-instruct':    [0.15, 0.6, 0],
                'Phi-3-medium-4k-instruct':  [0.17, 0.68, 0],
                'llama-3-8B':     [0.055, 0.055, 0],
                'llama-3-70B':    [0.35, 0.4, 0],
                'mixtral-8x7B':  [0.24, 0.24, 0]
                }

def price_of(q, model):
    # we assume the output size is constantly 1 for classification problems
    assert model in MODEL_PRICE, f'wrong: {model} not supported'
    # we simply apply the tiktoken to count tht token number for all models
    encoding = tiktoken.get_encoding('cl100k_base')
    in_token_num = len(encoding.encode(q))
    # for classification tasks, we assume the model always give out answer with len=1
    out_token_num = 1
    in_price = MODEL_PRICE[model][0] * in_token_num / 1e6
    out_price = MODEL_PRICE[model][1] * out_token_num / 1e6
    request_price = MODEL_PRICE[model][2]
    return in_price + out_price + request_price

def price_all(q, models):
    costs = []
    for m in models:
        costs.append(price_of(q, m))
    return costs
