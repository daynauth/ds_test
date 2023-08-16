import deepspeed
import torch 


import time
import argparse

device = 'cuda'



def get_deepspeed(model):
    ds_engine = deepspeed.init_inference(model, mp_size = 1, dtype = torch.float, replace_with_kernel_inject = True, checkpoint = None, replace_method = 'auto')
    return ds_engine

def get_model(name):
    if name == 'gpt2':
        from transformers import GPT2Tokenizer, GPT2Model
        model = GPT2Model.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return model, tokenizer
    elif name == 'gpt2-xl':
        from transformers import GPT2Tokenizer, GPT2Model
        model = GPT2Model.from_pretrained('gpt2-xl')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        return model, tokenizer
    elif name == 'bert-base':
        from transformers import BertTokenizer, BertModel
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return model, tokenizer
    elif name == 'bert-large':
        from transformers import BertTokenizer, BertModel
        model = BertModel.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        return model, tokenizer
    else:
        return None, None


def measure_inference(model, input_ids, num_iters = 1000):
    total_time = 0

    with torch.no_grad():
        for i in range(num_iters):
            start = time.time()
            output = model(input_ids)
            end = time.time()
            inference_time = end - start

            total_time += inference_time

    return total_time / num_iters * 1000


def run_test(model_name, ds = False):
    model, tokenizer = get_model(model_name)

    if ds:
        ds_engine = get_deepspeed(model)
        model = ds_engine.module
    else:
        model = model.to(device)

    input_text = "Hello, my dog is a"
    input_ids = tokenizer.encode(input_text, return_tensors = 'pt').to(device)

    #warmup
    output = model(input_ids)


    average_time =  measure_inference(model, input_ids)

    return average_time


def run_ds_test(model_name, ds = False):
    return run_test(model_name, ds = True)

def run_hf_test(model_name):
    return run_test(model_name, ds = False)

def main(model = 'gpt2', ds = False):
    if ds == True:
        average_time = run_ds_test(model)
        print(model, 'Average time for DeepSpeed: ', average_time, 'ms')
    else:
        average_time = run_hf_test(model)
        print(model, 'Average time for HuggingFace: ', average_time, 'ms')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type = str, default = 'gpt2', help = 'model name')
    argparser.add_argument('--ds', action="store_true", help = 'use deepspeed')
    argparser.add_argument("--local_rank", type=int, default=0)


    parser = deepspeed.add_config_arguments(argparser)
    args = parser.parse_args()

    model = args.model
    ds = args.ds

    main(model = model, ds = ds)

