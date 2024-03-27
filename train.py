import os
import sys

import torch
from tqdm import tqdm

from eval import contrastive_evaluation, info_nce_loss, standard_evaluation
from utils import build_dataloader


def supervised_training(
    model, train_dataset, val_dataset, loss_fn, device, args, silent=False
):
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
                f"checkpoints/{args['model']}_{args['dataset']}_{epoch_counter}.pt",
            )
            test_records.append(eval_res)

    torch.save(
        model.state_dict(),
        f"checkpoints/{args['model']}_{args['dataset']}_{epoch_counter}.pt",
    )
    if val_dataset is not None:
        eval_res = standard_evaluation(model, val_dataset, device, loss_fn, args)
        test_records.append(eval_res)
    return records, test_records


def contrastive_training(
    model, train_dataset, val_dataset, loss_fn, criterion, device, args, silent=False
):
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
        for images, _ in bar:
            images = torch.cat(images, dim=0)
            images = images.to(device)
            features = model(images)
            logits, labels = loss_fn(features, device, args)
            if torch.any(torch.nonzero(labels)):
                print("here")
            loss = criterion(logits, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            n_iter += 1
        if not silent:
            print(f"Epoch: {epoch_counter}\tLoss: {loss.item():.4f}\t")
        records.append({"epoch": epoch_counter, "loss": loss.item()})
        if epoch_counter >= 10:
            scheduler.step()

        if epoch_counter % args["log_every_n_steps"] == 0 and val_dataset is not None:
            eval_res = contrastive_evaluation(
                model, val_dataset, device, loss_fn, criterion, args
            )
            if not silent:
                sentence = f"Epoch: {epoch_counter}\t" + "\t".join(
                    [f"{k}: {v:.4f}" for k, v in eval_res.items()]
                )
                print(sentence)
            torch.save(
                model.state_dict(),
                f"checkpoints/{args['model']}_{args['dataset']}_{epoch_counter}.pt",
            )
            test_records.append(eval_res)

    torch.save(
        model.state_dict(),
        f"checkpoints/{args['model']}_{args['dataset']}_{epoch_counter}.pt",
    )
    if val_dataset is not None:
        eval_res = contrastive_evaluation(
            model, val_dataset, device, loss_fn, criterion, args
        )
        test_records.append(eval_res)
    return records, test_records


if __name__ == "__main__":
    import pandas as pd

    from dataset import SimCLRDataset
    from model import SimCLRCNN

    torch.manual_seed(4090)
    on_linux = sys.platform.startswith("linux")
    args = {
        "dataset": "cifar10",
        "model": "resnet50",
        "batch_size": 2048,
        "sample_rate": 1,
        "epochs": 1,
        "n_views": 2,
        "out_dim": 256,
        "lr": 3e-4,
        "wd": 1e-6,
        "log_every_n_steps": 5,
        "n_workers": 16,
        "temperature": 0.07,
        "learning": "contrastive",
        "val_split": 0,
        "ft_ratio": 0.1,
    }
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

    ft_train_records, ft_test_records = [], []

    if args["learning"] == "contrastive":
        loss_fn = info_nce_loss
        criterion = torch.nn.CrossEntropyLoss()
        train_records, val_records = contrastive_training(
            model, train_dataset, val_dataset, loss_fn, criterion, device, args
        )

        test_records = contrastive_evaluation(
            model, test_dataset, device, loss_fn, criterion, args
        )

    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        train_records, val_records = supervised_training(
            model, train_dataset, val_dataset, loss_fn, device, args
        )
        test_records = standard_evaluation(model, test_dataset, device, loss_fn, args)

    print("Test results: ", test_records)
    timestamp = pd.Timestamp.now().strftime("%m%d%H%M")
    torch.save(
        model.state_dict(), f"checkpoints/{args['model']}_{args['dataset']}_{timestamp}.pt",
    )

    pd.DataFrame.from_records(train_records).to_csv(
        f"logs/{args['model']}_{args['dataset']}_{timestamp}_train.csv", index=False
    )
    pd.DataFrame.from_records(val_records).to_csv(
        f"logs/{args['model']}_{args['dataset']}_{timestamp}_val.csv", index=False
    )
    if len(ft_train_records) > 0:
        pd.DataFrame.from_records(ft_train_records).to_csv(
            f"logs/{args['model']}_{args['dataset']}_{timestamp}_ft_train.csv",
            index=False,
        )
        pd.DataFrame.from_records(ft_test_records).to_csv(
            f"logs/{args['model']}_{args['dataset']}_{timestamp}_ft_val.csv",
            index=False,
        )
