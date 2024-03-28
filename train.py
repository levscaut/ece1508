import os
import sys
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval import contrastive_evaluation, info_nce_loss, standard_evaluation, info_nce_loss_newnew
from utils import build_dataloader


def supervised_training(
    model, train_dataset, val_dataset, loss_fn, device, args, artifacts_dir, silent=False
):
    if args['ckpt'] != "":
        model.load_state_dict(torch.load(args['ckpt']))
    ckpt_dir = f"{artifacts_dir}/checkpoints"
    log_dir = f"{artifacts_dir}/logs"
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)

    train_loader = build_dataloader(
        train_dataset, args["batch_size"], args["n_workers"]
    )
    os.makedirs("checkpoints", exist_ok=True)
    n_iter = 0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["wd"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )
    records = []
    test_records = []

    for epoch_counter in range(args["epochs"]):
        bar = train_loader
        if not silent:
            bar = tqdm(train_loader)
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.add_scalar("Train Loss", loss.item(), n_iter)
            n_iter += 1

        if not silent:
            print(f"Epoch: {epoch_counter}\tLoss: {loss.item():.4f}\t")
        records.append({"epoch": epoch_counter, "loss": loss.item()})
        if epoch_counter >= 10:
            scheduler.step()

        if epoch_counter % args["log_every_n_steps"] == 0 and val_dataset is not None:
            eval_res = standard_evaluation(model, val_dataset, device, loss_fn, args)
            if not silent:
                sentence = f"Epoch: {epoch_counter}\t" + "\t".join(
                    [f"{k}: {v:.4f}" for k, v in eval_res.items()]
                )
                print(sentence)
            torch.save(
                model.state_dict(),
                f"{ckpt_dir}/{epoch_counter}.pt",
            )
            test_records.append(eval_res)

    torch.save(
        model.state_dict(),
        f"{ckpt_dir}/final.pt",
    )
    if val_dataset is not None:
        eval_res = standard_evaluation(model, val_dataset, device, loss_fn, args)
        test_records.append(eval_res)
    return records, test_records


def contrastive_training(
    model, train_dataset, val_dataset, loss_fn, device, args, artifacts_dir, *, silent=False
):
    if args['ckpt'] != "":
        model.load_state_dict(torch.load(args['ckpt']))
    train_loader = build_dataloader(
        train_dataset, args["batch_size"], args["n_workers"]
    )

    log_dir = f"{artifacts_dir}/logs"
    ckpt_dir = f"{artifacts_dir}/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)

    n_iter = 0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["wd"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )
    records = []
    test_records = []

    for epoch_counter in range(args["epochs"]):
        bar = train_loader
        if not silent:
            bar = tqdm(train_loader)
        for images, _ in bar:
            images = torch.cat(images, dim=0)
            images = images.to(device)
            features = model(images)
            loss = loss_fn(features, device, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1
            logger.add_scalar("Loss", loss.item(), n_iter)
            logger.flush()
        if not silent:
            print(f"Epoch: {epoch_counter}\tLoss: {loss.item():.4f}\t")
        records.append({"epoch": epoch_counter, "loss": loss.item()})
        if epoch_counter >= 10:
            scheduler.step()

        if epoch_counter % args["log_every_n_steps"] == 0 and val_dataset is not None:
            eval_res = contrastive_evaluation(
                model, val_dataset, device, loss_fn, args
            )
            logger.add_scalar("Val Loss", eval_res["loss"], n_iter)
            if not silent:
                sentence = f"Epoch: {epoch_counter}\t" + "\t".join(
                    [f"{k}: {v:.4f}" for k, v in eval_res.items()]
                )
                print(sentence)
            test_records.append(eval_res)
        
        if epoch_counter % 10 * args["log_every_n_steps"] == 0:
            torch.save(
                model.state_dict(),
                f"{ckpt_dir}/{epoch_counter}.pt",
            )

    torch.save(
        model.state_dict(),
        f"{ckpt_dir}/final.pt",
    )
    if val_dataset is not None:
        eval_res = contrastive_evaluation(
            model, val_dataset, device, loss_fn, args
        )
        test_records.append(eval_res)
    return records, test_records

def parse_args():
    parser = argparse.ArgumentParser(description="Train a SimCLR model.")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name.")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--sample_rate", type=float, default=1, help="Sample rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--n_views", type=int, default=2, help="Number of views for contrastive learning.")
    parser.add_argument("--out_dim", type=int, default=256, help="Output dimension.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="Logging frequency.")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of workers.")
    parser.add_argument("--temperature", "-t", type=float, default=0.5, help="Temperature for contrastive loss.")
    parser.add_argument("--learning", type=str, default="contrastive", choices=["contrastive", "supervised"], help="Learning mode.")
    parser.add_argument("--val_split", type=float, default=0.0, help="Validation split ratio.")
    parser.add_argument("--ft_ratio", type=float, default=0.1, help="Fine-tuning ratio.")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint resume path")
    return vars(parser.parse_args())


def train(args):
    import pandas as pd

    from dataset import SimCLRDataset
    from model import SimCLRCNN

    torch.manual_seed(4090)
    timestamp = pd.Timestamp.now().strftime("%m%d%H%M")
    on_linux = sys.platform.startswith("linux")
    artifacts_dir = f"artifacts/{args['model']}_{args['dataset']}_{timestamp}"
    os.makedirs(artifacts_dir, exist_ok=True)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = SimCLRDataset(args["dataset"])

    train_dataset, val_dataset = data.get_train_val_datasets(
        args["n_views"], args["val_split"]
    )
    test_dataset = data.get_test_dataset(args["n_views"])
    num_classes = data.num_classes

    model_args = {
        "backbone": args["model"],
        "out_dim": args["out_dim"]
        if args["learning"] == "contrastive"
        else num_classes,
        "mod": args["learning"] == "contrastive",
    }
    model = SimCLRCNN(**model_args).to(device)

    if on_linux:
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")

    if args["learning"] == "contrastive":
        loss_fn = info_nce_loss_newnew
        train_records, val_records = contrastive_training(
            model, train_dataset, val_dataset, loss_fn, device, args, artifacts_dir
        )

        test_records = contrastive_evaluation(
            model, test_dataset, device, loss_fn, args
        )

    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        train_records, val_records = supervised_training(
            model, train_dataset, val_dataset, loss_fn, device, args, artifacts_dir
        )
        test_records = standard_evaluation(model, test_dataset, device, loss_fn, args)

    print("Test results: ", test_records)


    pd.DataFrame(train_records).to_csv(
        f"{artifacts_dir}/train.csv", index=False
    )
    pd.DataFrame(val_records).to_csv(
        f"{artifacts_dir}/val.csv", index=False
    )
    pd.DataFrame.from_dict(test_records, orient="index").T.to_csv(
        f"{artifacts_dir}/test.csv", index=False
    )
    return test_records


if __name__ == "__main__":
    args = parse_args()
    train(args)
    """
    # Hyperparameter tuning example

    import flaml.tune
    
    args['lr'] = flaml.tune.loguniform(1e-4, 1)
    args['epochs'] = 10
    analysis = flaml.tune.run(
        train,
        config=args,
        num_samples=10,
        metric="loss",
        mode="min",
    )
    print(analysis.best_config)
    """
