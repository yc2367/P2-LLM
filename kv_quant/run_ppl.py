import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
from utils import load_model_and_tokenizer, add_common_args, add_cache_args, build_cache
from loguru import logger

def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        # Wrapper for tokenized input IDs
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        return testenc
    else:
        raise NotImplementedError
    

def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        # Wrapper for tokenized input IDs
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        #testenc = tokenizer("\n\n".join(valdata["text"]), return_tensors="pt")
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        return testenc
    else:
        raise NotImplementedError

@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda"):
    model = model.to(device)
    if isinstance(device, str):
        device = torch.device(device)

    results = {}

    for dataset in datasets.split(","):
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all_seqlen_{seqlen}.cache"
        )
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
        else:
            testloader = get_ppl_eval_loaders(dataset, tokenizer, seqlen=seqlen)
            torch.save(testloader, cache_testloader)
        
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        
        #NOTE(brian1009): Comment out the cache for now
        #model.config.use_cache = False
        model.eval()

        nlls = []

        for i in tqdm(range(nsamples)):
            cache = build_cache(args, device, model.dtype, for_ppl_test=True)   
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(
                    device
                )
            outputs = model.model(batch, past_key_values=cache)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
            del cache
            
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        model.config.use_cache = use_cache
        results.update({dataset: ppl.item()})

    return results
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_cache_args(parser)
    parser.add_argument('--datasets', type=str, help='datasets to evaluate', default='wikitext2')
    parser.add_argument('--seqlen', type=int, help='sequence length for ppl evaluation', default=2048)
    parser.add_argument("--device", type=str, help="device to run the model on", default="cuda")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    
    logger.info(f"Start evaluating ppl...")
    logger.info(f"*model: {args.model_name_or_path}")
    logger.info(f"*datasets: {args.datasets}")
    logger.info(f"*sequence length {args.seqlen}")
    results = eval_ppl(model, tokenizer, args.model_name_or_path, args.datasets, args.seqlen, args.device)
    for dataset, ppl in results.items():
        logger.info(f"PPL: {ppl}")