import sys
import torch

from lib.utils import (
    parse_arguments,
    load_model,
    predict,
    print_results,
    store_results,
)

def main():
    args = parse_arguments()
    if args.task not in ['haystack', 'form', 'issue', 'target', 'multi']:
        print("User error: --task must be either " +\
                        "haystack, form, issue, target or multi.")
        sys.exit(0)

    # Read the article as string into variable
    with open(args.article, 'r') as file:
        text = file.read().replace('\n', '')

    model, tokenizer = load_model(args)

    # Tokenize article as input to model
    text_tokenized = torch.tensor([tokenizer.encode(text)])

    results = predict(model, text_tokenized, args)
    print_results(results, args)
    if args.output_path != '':
        store_results(results, args)


if __name__ == "__main__":
    main()
