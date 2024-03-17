import pandas as pd
import os
import sys

import torch
from tqdm import tqdm

from eval import contrastive_evaluation, info_nce_loss, standard_evaluation
from train import contrastive_training, supervised_training
from dataset import SimCLRDataset
from model import SimCLRCNN

def train(args):
    on_linux = sys.platform.startswith("linux")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = SimCLRDataset(args["dataset"])
    build_dataloader = lambda dataset: torch.utils.data.DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=args["n_workers"],
    )

    train_dataset, val_dataset = data.get_train_val_datasets(
        args["n_views"], args["val_split"]
    )
    train_loader = build_dataloader(train_dataset)
    val_loader = build_dataloader(val_dataset)
    test_dataset = data.get_test_dataset(args["n_views"])
    test_loader = build_dataloader(test_dataset)
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
        loss_fn = info_nce_loss
        criterion = torch.nn.CrossEntropyLoss()
        train_records, test_records = contrastive_training(
            model, train_loader, val_loader, loss_fn, criterion, device, args, silent=True
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        train_records, test_records = supervised_training(
            model, train_loader, val_loader, loss_fn, device, args, silent=True
        )

    best_top5 = max([record["top5"] for record in test_records])
    print(f"Best top5: {best_top5}")
    return {"top5": best_top5}


if __name__ == "__main__":
    from flaml import tune
    import pickle
    args = {
        "dataset": "cifar10",
        "model": "resnet50",
        "batch_size": 256,
        "sample_rate": 1,
        "epochs": 100,
        "n_views": 2,
        "out_dim": tune.choice([128, 256]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "wd": 1e-6,
        "log_every_n_steps": 5,
        "n_workers": 16,
        "temperature": 0.07,
        "learning": "contrastive",
        "val_split": 0.2,
    }
    res = tune.run(train, config=args, metric="top5", mode="max", num_samples=10)
    pickle.dump(res, open("tune_res.pkl", "wb"))
