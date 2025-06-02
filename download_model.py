import os 
import sys 
from copy import deepcopy 
from argparse import Namespace, ArgumentParser 
import yaml

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoConfig,
    AutoModel, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AwqConfig 
)

if __name__ == "__main__": 

    parser = ArgumentParser(description="args") 
    parser.add_argument("model_name_or_path", type=str, help="huggingface id") 
    parser.add_argument("--type", type=str, default="causal", choices=[None, "causal", "seq2seq"], help="type of model")
    parser.add_argument("--save_to", type=str, default=None, help="Directory to save the model to.")
    parser.add_argument("--quant_config", type=str, default=None, help="path to quantization YAML file (LLMC)")

    args = parser.parse_args()

    # Manual override of args 
    args.quant_config = "configs/custom/awq_w_a.yml"


    if args.type == "causal": 
        AutoClass = AutoModelForCausalLM 
    elif args.type == "seq2seq": 
        AutoClass = AutoModelForSeq2SeqLM
    else: 
        AutoClass = AutoModel
        
    
    # This does not work right now, get rid of it
    if args.quant_config is not None: 
        with open(args.quant_config, "r") as f: 
            data = yaml.safe_load(f)
            method = data["quant"]["method"]

        if method == "Awq":  
            weight = data["quant"]["weight"]
            quant_config = AwqConfig(bits=weight["bit"], 
                                     group_size=weight["group_size"])
        else: 
            raise NotImplementedError
    else: 
        quant_config = None





    model = AutoClass.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
                                    #   quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


    save_dest = os.path.join(args.save_to, args.model_name_or_path) if args.save_to is not None else args.model_name_or_path

    model.save_pretrained(save_dest)
    tokenizer.save_pretrained(save_dest)
