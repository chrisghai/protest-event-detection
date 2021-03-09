import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import trange
from pprint import pprint, pformat
from argparse import ArgumentParser

from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

from lib.models import (
    RobertaProtestAuxClassification,
    RobertaMultiTaskClassification,
)

MODEL_FOLDER = 'models'
MODEL_FOLDER = '/home/chris/Downloads/models'
HAYSTACK_CLASSES = {
    0: 'Non-protest', 
    1: 'Protest',
}

FORM_CLASSES = {
    0: 'Blockade/slowdown/disruption', 
    1: 'Boycott',
    2: 'Hunger strike', 
    3: 'March', 
    4: 'Non-protest',
    5: 'Rally/demonstration', 
    6: 'Riot',
    7: 'Strike/walkout/lockout',
}

ISSUE_CLASSES = {
    0: 'Anti-colonial/political independence',
    1: 'Anti-war/peace', 
    2: 'Criminal justice system',
    3: 'Democratisation', 
    4: 'Economy/inequality',
    5: 'Environmental', 
    6: 'Foreign policy',
    7: 'Human and civil rights', 
    8: 'Labour & work',
    9: 'Non-protest', 
    10: 'Political corruption/malfeasance',
    11: 'Racial/ethnic rights', 
    12: 'Religion',
    13: 'Social services & welfare', 
    14: 'None of the above',
}

TARGET_CLASSES = {
    0: 'Domestic government', 
    1: 'Foreign government',
    2: 'Individual', 
    3: 'Intergovernmental organisation',
    4: 'Non-protest', 
    5: 'Private/business',
}

TASK_DICT = {
    'form': FORM_CLASSES,
    'issue': ISSUE_CLASSES,
    'target': TARGET_CLASSES,
}

TASK_LENGTHS = {
    'form': 8, 
    'issue': 15, 
    'target': 6,
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--article', action='store', type=str,
            help="Path to file with text to classify.")
    parser.add_argument('--output_path', action='store', type=str, default='',
            help="Path to location where the prediction will be written to file.")
    parser.add_argument('--out_file', action='store', type=str,
            help="Filename where the prediction will be stored.", default='pred.txt')
    parser.add_argument('--task', type=str, default='haystack',
            help="Set the task type: haystack, form, issue, target or multi (all).")
    parser.add_argument('--mc_samples', action='store', type=int, default=0,
            help="Number of Monte Carlo samples.")
    parser.add_argument('--gpu_devices', type=int, default=[0], nargs='+',
            help="Number of GPU(s) to use.")
    args = parser.parse_args()
    return args


def load_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = RobertaConfig.from_pretrained(f'{MODEL_FOLDER}/haystack/{device}/')
    if args.task == 'haystack':
        config = RobertaConfig.from_pretrained(f'{MODEL_FOLDER}/haystack/{device}/')
        model = RobertaForSequenceClassification.from_pretrained(
            f'{MODEL_FOLDER}/haystack/{device}/',
            config=config,
        )

    elif args.task in ['form', 'issue', 'target']:
        config.aux_classes = TASK_LENGTHS[args.task]
        model = RobertaProtestAuxClassification.from_pretrained(
            f'{MODEL_FOLDER}/{args.task}/{device}/',
            config=config,
        )

    else:
        config.form_classes     = TASK_LENGTHS['form']
        config.issue_classes    = TASK_LENGTHS['issue']
        config.target_classes   = TASK_LENGTHS['target']
        model = RobertaMultiTaskClassification.from_pretrained(
            f'{MODEL_FOLDER}/multi/cuda/',
            config=config
        )

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(id) for id in args.gpu_devices)
        model.cuda()
    else:
        model.cpu()

    tokenizer = RobertaTokenizer.from_pretrained(
        f'{MODEL_FOLDER}/haystack/{device}'
    )

    return model, tokenizer


def predict(model, inputs, args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(id) for id in args.gpu_devices)
        model.cuda()
        inputs = inputs.to("cuda")
    else:
        model.cpu()
        inputs = inputs.to("cpu")

    if args.mc_samples > 0:
        model.train()
    else:
        model.eval()

    n = max(1, args.mc_samples)
    for m in trange(n, desc='Iteration'):
        with torch.no_grad():
            output = model(inputs)

            haystack_dist = nn.functional.softmax(output[0], dim=-1)
            if args.task in ['form', 'issue', 'target']:
                aux_dist = nn.functional.softmax(output[1], dim=-1)
            elif args.task == 'multi':
                form_dist = nn.functional.softmax(output[1], dim=-1)
                issue_dist = nn.functional.softmax(output[2], dim=-1)
                target_dist = nn.functional.softmax(output[3], dim=-1)

        if m == 0:
            haystack_mat = haystack_dist.detach().cpu().numpy()
            if args.task in ['form', 'issue', 'target']:
                aux_mat = aux_dist.detach().cpu().numpy()
            elif args.task == 'multi':
                form_mat = form_dist.detach().cpu().numpy()
                issue_mat = issue_dist.detach().cpu().numpy()
                target_mat = target_dist.detach().cpu().numpy()

        else:
            haystack_mat = np.concatenate(
                (haystack_mat, haystack_dist.detach().cpu().numpy())
            )

            if args.task in ['form', 'issue', 'target']:
                aux_mat = np.concatenate(
                    (aux_mat, aux_dist.detach().cpu().numpy())
                )
            elif args.task == 'multi':
                form_mat = np.concatenate(
                    (form_mat, form_dist.detach().cpu().numpy())
                )
                issue_mat = np.concatenate(
                    (issue_mat, issue_dist.detach().cpu().numpy())
                )
                target_mat = np.concatenate(
                    (target_mat, target_dist.detach().cpu().numpy())
                )

    results = {}
    if args.mc_samples > 0:
        haystack_std_err = [haystack_mat[:, 0].std(), haystack_mat[:, 1].std()]
        haystack_mean = haystack_mat.mean(axis=0)
        haystack_entropy = -np.sum(haystack_mean * np.log2(haystack_mean))
        haystack_prediction = np.argmax(haystack_mean).flatten()
        haystack_prediction = np.vectorize(HAYSTACK_CLASSES.get)(
            haystack_prediction.astype(int)
        )
        results['haystack_prediction'] = haystack_prediction
        results['haystack_std_err']    = haystack_std_err
        results['haystack_entropy']    = haystack_entropy
        results['haystack_mean']       = haystack_mean

        if args.task in ['form', 'issue', 'target']:
            aux_std_err = [aux_mat[:, i].std() for i in range(TASK_LENGTHS[args.task])]
            aux_mean = aux_mat.mean(axis=0)
            aux_entropy = -np.sum(aux_mean * np.log2(aux_mean))
            aux_prediction = np.argmax(aux_mean).flatten()
            aux_prediction = np.vectorize(TASK_DICT[args.task].get)(
                aux_prediction.astype(int)
            )
            results[f'{args.task}_prediction'] = aux_prediction
            results[f'{args.task}_std_err']    = aux_std_err
            results[f'{args.task}_entropy']    = aux_entropy
            results[f'{args.task}_mean']       = aux_mean

        elif args.task == 'multi':
            form_std_err = [form_mat[:, i].std() for i in range(TASK_LENGTHS['form'])]
            form_mean = form_mat.mean(axis=0)
            form_entropy = -np.sum(form_mean * np.log2(form_mean))
            form_prediction = np.argmax(form_mean).flatten()
            form_prediction = np.vectorize(FORM_CLASSES.get)(
                form_prediction.astype(int)
            )
            results['form_prediction'] = form_prediction
            results['form_std_err']    = form_std_err
            results['form_entropy']    = form_entropy
            results['form_mean']       = form_mean

            issue_std_err = [issue_mat[:, i].std() for i in range(TASK_LENGTHS['issue'])]
            issue_mean = issue_mat.mean(axis=0)
            issue_entropy = -np.sum(issue_mean * np.log2(issue_mean))
            issue_prediction = np.argmax(issue_mean).flatten()
            issue_prediction = np.vectorize(ISSUE_CLASSES.get)(
                issue_prediction.astype(int)
            )
            results['issue_prediction'] = issue_prediction
            results['issue_std_err']    = issue_std_err
            results['issue_entropy']    = issue_entropy
            results['issue_mean']       = issue_mean

            target_std_err = [target_mat[:, i].std() for i in range(TASK_LENGTHS['target'])]
            target_mean = target_mat.mean(axis=0)
            target_entropy = -np.sum(target_mean * np.log2(target_mean))
            target_prediction = np.argmax(target_mean).flatten()
            target_prediction = np.vectorize(TARGET_CLASSES.get)(
                target_prediction.astype(int)
            )
            results['target_prediction'] = target_prediction
            results['target_std_err']    = target_std_err
            results['target_entropy']    = target_entropy
            results['target_mean']       = target_mean

    else:
            haystack_prediction = np.argmax(haystack_mat).flatten()
            haystack_prediction = np.vectorize(HAYSTACK_CLASSES.get)(
                haystack_prediction.astype(int)
            )
            results['haystack_prediction'] = haystack_prediction

            if args.task in ['form', 'issue', 'target']:
                aux_prediction = np.argmax(aux_mat).flatten()
                aux_prediction = np.vectorize(TASK_DICT[args.task].get)(
                    aux_prediction.astype(int)
                )
                results[f'{args.task}_prediction'] = aux_prediction

            elif args.task == 'multi':
                form_prediction = np.argmax(form_mat).flatten()
                form_prediction = np.vectorize(FORM_CLASSES.get)(
                    form_prediction.astype(int)
                )
                results['form_prediction'] = form_prediction

                issue_prediction = np.argmax(issue_mat).flatten()
                issue_prediction = np.vectorize(ISSUE_CLASSES.get)(
                    issue_prediction.astype(int)
                )
                results['issue_prediction'] = issue_prediction

                target_prediction = np.argmax(target_mat).flatten()
                target_prediction = np.vectorize(TARGET_CLASSES.get)(
                    target_prediction.astype(int)
                )
                results['target_prediction'] = target_prediction

    return results


def print_results(results, args):
    print(f"{'='*15} RESULTS {'=' * 15}")
    print(f"Haystack prediction: {results['haystack_prediction']}")
    if args.mc_samples > 0:
        print(f"Haystack mean probabilities: {np.round(results['haystack_mean'], 6)}")
        print(f"Haystack standard errors: {np.round(results['haystack_std_err'], 6)}")
        print(f"Haystack entropy: {np.round(results['haystack_entropy'], 6)}")
        pprint(HAYSTACK_CLASSES)

    print(f"{'=' * 40}")
    if args.task in ['form', 'issue', 'target']:
        print(f"{args.task.title()} prediction: {results[f'{args.task}_prediction']}")
        if args.mc_samples > 0:
            print(f"{args.task.title()} mean probabilities: {np.round(results[f'{args.task}_mean'], 6)}")
            print(f"{args.task.title()} standard errors: {np.round(results[f'{args.task}_std_err'], 6)}")
            print(f"{args.task.title()} entropy: {np.round(results[f'{args.task}_entropy'], 6)}")
            pprint(TASK_DICT[args.task])
        print(f"{'=' * 40}")

    elif args.task == 'multi':
        print(f"Form prediction: {results['form_prediction']}")
        if args.mc_samples > 0:
            print(f"Form mean probabilities: {np.round(results['form_mean'], 6)}")
            print(f"Form standard errors: {np.round(results['form_std_err'], 6)}")
            print(f"Form entropy: {np.round(results['form_entropy'], 6)}")
            pprint(FORM_CLASSES)
        print(f"{'=' * 40}")

        print(f"Issue prediction: {results['issue_prediction']}")
        if args.mc_samples > 0:
            print(f"Issue mean probabilities: {np.round(results['issue_mean'], 6)}")
            print(f"Issue standard errors: {np.round(results['issue_std_err'], 6)}")
            print(f"Issue entropy: {np.round(results['issue_entropy'], 6)}")
            pprint(ISSUE_CLASSES)
        print(f"{'=' * 40}")

        print(f"Target prediction: {results['target_prediction']}")
        if args.mc_samples > 0:
            print(f"Target mean probabilities: {np.round(results['target_mean'], 6)}")
            print(f"Target standard errors: {np.round(results['target_std_err'], 6)}")
            print(f"Target entropy: {np.round(results['target_entropy'], 6)}")
            pprint(TARGET_CLASSES)
        print(f"{'=' * 40}")


def store_results(results, args):
    output_path = args.output_path
    if not output_path.endswith('/'):
        output_path += '/'

    out_file = args.out_file
    if not out_file.endswith('.txt'):
        output_file += '.txt'

    with open(f"{output_path}{out_file}", 'w+') as f:
        f.write(f"{'='*15} RESULTS {'=' * 15}\n")
        f.write(f"Haystack prediction: {results['haystack_prediction']}\n")
        if args.mc_samples > 0:
            f.write(f"Haystack mean probabilities: {np.round(results['haystack_mean'], 6)}\n")
            f.write(f"Haystack standard errors: {np.round(results['haystack_std_err'], 6)}\n")
            f.write(f"Haystack entropy: {np.round(results['haystack_entropy'], 6)}\n")
            f.write(f"{pformat(HAYSTACK_CLASSES, indent=4)}\n")

        f.write(f"{'=' * 40}\n")
        if args.task in ['form', 'issue', 'target']:
            f.write(f"{args.task.title()} prediction: {results[f'{args.task}_prediction']}\n")
            if args.mc_samples > 0:
                f.write(f"{args.task.title()} mean probabilities: {np.round(results[f'{args.task}_mean'], 6)}\n")
                f.write(f"{args.task.title()} standard errors: {np.round(results[f'{args.task}_std_err'], 6)}\n")
                f.write(f"{args.task.title()} entropy: {np.round(results[f'{args.task}_entropy'], 6)}\n")
                f.write(f"{pformat(TASK_DICT[args.task], indent=4)}\n")
            f.write(f"{'=' * 40}\n")

        elif args.task == 'multi':
            f.write(f"Form prediction: {results['form_prediction']}\n")
            if args.mc_samples > 0:
                f.write(f"Form mean probabilities: {np.round(results['form_mean'], 6)}\n")
                f.write(f"Form standard errors: {np.round(results['form_std_err'], 6)}\n")
                f.write(f"Form entropy: {np.round(results['form_entropy'], 6)}\n")
                f.write(f"{pformat(FORM_CLASSES, indent=4)}\n")
            f.write(f"{'=' * 40}\n")

            f.write(f"Issue prediction: {results['issue_prediction']}\n")
            if args.mc_samples > 0:
                f.write(f"Issue mean probabilities: {np.round(results['issue_mean'], 6)}\n")
                f.write(f"Issue standard errors: {np.round(results['issue_std_err'], 6)}\n")
                f.write(f"Issue entropy: {np.round(results['issue_entropy'], 6)}\n")
                f.write(f"{pformat(ISSUE_CLASSES, indent=4)}\n")
            f.write(f"{'=' * 40}\n")

            f.write(f"Target prediction: {results['target_prediction']}\n")
            if args.mc_samples > 0:
                f.write(f"Target mean probabilities: {np.round(results['target_mean'], 6)}\n")
                f.write(f"Target standard errors: {np.round(results['target_std_err'], 6)}\n")
                f.write(f"Target entropy: {np.round(results['target_entropy'], 6)}\n")
                f.write(f"{pformat(ISSUE_CLASSES, indent=4)}\n")
            f.write(f"{'=' * 40}\n")
