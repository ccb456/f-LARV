import argparse
import os

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
from tqdm import tqdm


def train(args, model, train_features, dev_features, test_features):

    scaler = GradScaler()
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            print(f"Epoch {epoch + 1} has started !")
            model.zero_grad()
            for step, batch in tqdm(enumerate(train_dataloader)):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs[0] / args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    print({"loss": loss.item()})
                    dev_score, dev_output, dev_recall = evaluate(args, model, dev_features, tag="dev")
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        print("save result")
                        # pred = report(args, model, test_features)
                        # with open("result.json", "w") as fh:
                        #     json.dump(pred, fh)
                        if args.save_path != "":
                            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                            torch.save(model.state_dict(), args.save_path)
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)
    print('training finished')


def evaluate(args, model, features, tag="dev"):

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
    ans = to_official(preds, features)
    best_f1 = 0
    if len(ans) > 0:
        best_f1, re_p, re_r = official_evaluate(ans, args.data_dir)
    output = {
        "delimiter": '*' * 50,
        tag + "_F1": best_f1 * 100,
        tag + "_P": re_p * 100,
        tag + "_recall": re_r * 100,
    }
    return best_f1, output, re_r


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




def generate_step1_json(
    args,
    model,
    test_features,
    output_path,
    threshold=0.27
):
    """
    Stage 1: RCP
    Generate step1.json for test set
    """

    dataloader = DataLoader(
        test_features,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    model.eval()
    step1_results = []

    doc_ptr = 0  # 指向 test_features 中当前文档

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'entity_pos': batch[3],
                'hts': batch[4],          # List[List[[h,t]]]
            }

            batch_hts = batch[4]          # doc-level entity pairs

            # -------- forward --------
            logits, *_ = model(**inputs)  # (num_pairs_in_batch, 2)
            probs = torch.softmax(logits, dim=1)[:, 1]  # have_relation prob
            probs = probs.cpu()

            # -------- split back to document level --------
            offset = 0
            for doc_hts in batch_hts:
                n_pairs = len(doc_hts)
                doc_probs = probs[offset: offset + n_pairs]
                offset += n_pairs

                kept_pairs = [
                    pair for pair, p in zip(doc_hts, doc_probs)
                    if p > threshold
                ]

                # 从 test_features 中取 title（最稳妥）
                title = test_features[doc_ptr]["title"]
                doc_ptr += 1

                step1_results.append({
                    "title": title,
                    "entity_pairs": kept_pairs
                })

    # -------- save --------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(step1_results, f, ensure_ascii=False, indent=2)

    print(f"[RCP] step1.json saved to {output_path}")
    print(f"[RCP] threshold = {threshold}")
    print(f"[RCP] total kept pairs = {sum(len(x['entity_pairs']) for x in step1_results)}")




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../datasets/redocred", type=str)
    parser.add_argument("--transformer_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="../PLM/roberta-large", type=str)

    parser.add_argument("--train_file", default="train_revised.json", type=str)
    parser.add_argument("--dev_file", default="dev_revised.json", type=str)
    parser.add_argument("--test_file", default="test_revised.json", type=str)
    parser.add_argument("--save_path", default="../model/Re-DocRED/BCE-5/redocred.pt", type=str)
    parser.add_argument("--step_save_dir", default="", type=str)
    # parser.add_argument("--load_path", default="../model/Re-DocRED/Foal-0.2-recall-best/redocred.pt", type=str)
    parser.add_argument("--load_path", default="../model/Re-DocRED/Focal-0.2-F1-best/redocred.pt", type=str)
    
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

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

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
        # pass
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path))
        # dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        # print(dev_output)
        # pred = report(args, model, test_features)
        # with open("./4090/best-recall/0.27.json", "w") as fh:
        #     json.dump(pred, fh)
        model.to(args.device)

        generate_step1_json(
            args=args,
            model=model,
            test_features=dev_features,
            output_path="results/redocred/dev-best-F1-0.3.json",
            threshold=0.3
        )


if __name__ == "__main__":
    main()
