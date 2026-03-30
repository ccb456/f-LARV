import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def evaluate_rcp_on_dev(args, model, dev_features, threshold):
    dev_dataloader = DataLoader(
        dev_features,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dev_dataloader:
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'entity_pos': batch[3],   # list (doc-level)
                'hts': batch[4],          # list (doc-level)
            }

            # -------- forward --------
            logits, *_ = model(**inputs)
            # logits shape: (num_pairs, 2)
            if logits.dim() != 2 or logits.size(1) != 2:
                raise ValueError(f"Expected logits shape (N,2), got {logits.shape}")

            # ★ 正确的二分类概率：softmax
            probs = torch.softmax(logits, dim=1)[:, 1]   # (num_pairs,)

            # -------- labels: flatten to pair-level --------
            raw_labels = batch[2]   # List[List[[no_rel, have_rel]]]

            flat_labels = []
            for doc_labels in raw_labels:
                for rel in doc_labels:
                    flat_labels.append(rel[1])

            labels = torch.tensor(
                flat_labels,
                dtype=torch.long,
                device=args.device
            )

            if labels.numel() != probs.size(0):
                raise ValueError(
                    f"Label/prob mismatch: labels={labels.numel()}, probs={probs.size(0)}"
                )

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # -------- RCP gating --------
    # print(
    #     "probs min / mean / max:",
    #     all_probs.min().item(),
    #     all_probs.mean().item(),
    #     all_probs.max().item()
    # )

    preds = (all_probs > threshold).long()

    tp = ((preds == 1) & (all_labels == 1)).sum().item()
    fn = ((preds == 0) & (all_labels == 1)).sum().item()

    kept_pairs = preds.sum().item()
    total_pairs = all_labels.numel()

    recall = tp / (tp + fn + 1e-8)

    return {
        "threshold": threshold,
        "recall": recall,
        "kept_pairs": kept_pairs,
        "retention_ratio": kept_pairs / total_pairs,
    }




def grid_search_rcp_threshold(args, model, dev_features, thresholds):
    results = []

    for th in thresholds:
        metrics = evaluate_rcp_on_dev(
            args=args,
            model=model,
            dev_features=dev_features,
            threshold=th
        )
        results.append(metrics)

        print(
            f"th={th:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"kept={metrics['kept_pairs']} | "
            f"retention={metrics['retention_ratio']:.4f}"
        )

    return results




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../datasets/redocred", type=str)
    parser.add_argument("--transformer_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="../PLM/roberta-large", type=str)

    parser.add_argument("--train_file", default="train_revised.json", type=str)
    parser.add_argument("--dev_file", default="dev_revised.json", type=str)
    parser.add_argument("--test_file", default="test_revised.json", type=str)
    parser.add_argument("--save_path", default="../model/redocred.pt", type=str)
    parser.add_argument("--step_save_dir", default="", type=str)
    # parser.add_argument("--load_path", default="../model/Re-DocRED/Focal-0.2-F1-best/redocred.pt", type=str)
    parser.add_argument("--load_path", default="../model/Re-DocRED/Foal-0.2-recall-best/redocred.pt", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=200, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    dev_file = os.path.join(args.data_dir, args.dev_file)
    # test_file = os.path.join(args.data_dir, args.test_file)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    # test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)


    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model)
    model.to(0)

    if args.load_path == "":  # Training
        pass
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path))
        thresholds = [0.270, 0.275, 0.280, 0.285, 0.290, 0.295, 0.300, 0.305, 0.310, 0.315, 0.320, 0.330, 0.340]
        results = grid_search_rcp_threshold(args, model, dev_features, thresholds)


if __name__ == "__main__":
    main()
