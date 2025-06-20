import os
from argparse import ArgumentParser, Namespace

import datasets 
from datasets import load_dataset 

parser = ArgumentParser("Download dataset")
parser.add_argument("name", choices=["wikitext", "c4", "pileval", "hellaswag"])

args = parser.parse_args()

if args.name == "wikitext": 
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    train.save_to_disk("wikitext_local/train")
    val.save_to_disk("wikitext_local/validation")
    test.save_to_disk("wikitext_local/test")
elif args.name == "c4": 
    testdata = load_dataset(
                        'allenai/c4',
                        data_files={
                            'validation': 'en/c4-validation.00000-of-00008.json.gz'
                        },
                        split='validation',
                    )

    testdata.save_to_disk("c4_local/validation")
elif args.name == "pileval": 
    valset = load_dataset('mit-han-lab/pile-val-backup', split='validation')
    valset.save_to_disk("pileval_local/validation")

elif args.name == "hellaswag": 
    valset = load_dataset("hellaswag", split="validation")
    trainset = load_dataset("hellaswag", split="train")

    valset.save_to_disk("hellaswag_local/validation")
    trainset.save_to_disk("hellaswag_local/train")
else: 
    raise NotImplementedError()