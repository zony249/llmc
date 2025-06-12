from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json


from vllm import LLM, SamplingParams
from accelerate import (
    infer_auto_device_map,
    dispatch_model,
)
from accelerate.utils.modeling import get_balanced_memory
from lm_eval_adaptor import LMEvalAdaptor
from datasets import load_dataset
from torch import nn
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
args = parser.parse_args()

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}







def build_model_and_enc(model_path, dtype):
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")
    # all hf model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
    config.use_cache = False
    enc = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    # Init model on CPU:
    kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    # if not vila_10_quant_mode:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs
    )
    model.eval()

    kwargs = {
        "max_memory": get_balanced_memory(
            model, max_memory if len(max_memory) > 0 else None
        )
    }
    device_map = infer_auto_device_map(
        model,
        # TODO: can we remove this?
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "BloomBlock",
            "MPTBlock",
            "DecoderLayer",
        ],
        **kwargs,
    )
    model = dispatch_model(model, device_map=device_map)

    return model, enc






def main():



    print(args)
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    model, enc = build_model_and_enc(args.model_path, args.dtype)

    if args.tasks is not None:

        task_names = args.tasks.split(",")
        non_special_tasks = []
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        for task in task_names:
            if task in ["wikitext", "c4"]:
                if task == "wikitext":
                    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
                elif task == "c4": 
                    testenc = load_dataset(
                                'allenai/c4',
                                data_files={
                                    'validation': 'en/c4-validation.00000-of-00008.json.gz'
                                },
                                split='validation',
                            )
                    testenc = enc(" ".join(testenc[:1100]["text"]), return_tensors="pt")


                # print("ENC:", enc)
                model.seqlen = 2048
                testenc = testenc.input_ids.to(model.device)
                nsamples = testenc.numel() // model.seqlen
                model = model.eval()
                nlls = []
                for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                    batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                        model.device
                    )
                    with torch.no_grad():
                        # lm_logits = model(batch).logits
                        lm_logits = model(batch).logits

                    shift_logits = lm_logits[:, :-1, :].contiguous().float()
                    shift_labels = testenc[
                        :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                    ][:, 1:]
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )
                    neg_log_likelihood = loss.float() * model.seqlen
                    nlls.append(neg_log_likelihood)

                ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
                print(ppl.item())

                results = {"ppl": ppl.item()}
                if args.output_path is not None:
                    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                    with open(args.output_path, "w") as f:
                        json.dump(results, f, indent=2)
            else:
                non_special_tasks.append(task)  


        if len(non_special_tasks) > 0 :

            lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=non_special_tasks,
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=args.num_fewshot,
            )

            print(evaluator.make_table(results))

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = args.model_path
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
