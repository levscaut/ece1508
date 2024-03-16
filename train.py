from tqdm import tqdm
import os
import sys
import torch
from eval import standard_evaluation, contrastive_evaluation, info_nce_loss

def supervised_training(model, train_loader, val_loader, loss_fn, device, args):
    os.makedirs("checkpoints", exist_ok=True)
    n_iter = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    records = []
    test_records = []

    for epoch_counter in range(args['epochs']):

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            n_iter += 1
        
        print(f"Epoch: {epoch_counter}\tLoss: {loss.item():.4f}\t")
        records.append({"epoch": epoch_counter, "loss": loss.item()})
        if epoch_counter >= 10:
            scheduler.step()

        if epoch_counter % args["log_every_n_steps"] == 0:
            eval_res = standard_evaluation(model, val_loader, device, loss_fn, args)
            top1, top5 = eval_res["top1"], eval_res["top5"]
            test_loss = eval_res["loss"]
            print(f"Epoch: {epoch_counter}\tTop1 accuracy: {top1:.4f}\tTop5 accuracy: {top5:.4f}\tTest loss: {test_loss:.4f}")
            torch.save(model.state_dict(), f"checkpoints/{args['model']}_{args['dataset']}_{epoch_counter}.pt")
            test_records.append(eval_res)

    print("Training has finished.")
    torch.save(model.state_dict(), f"checkpoints/{args['model']}_{args['dataset']}_{epoch_counter}.pt")
    return records, test_records

def contrastive_training(model, train_loader, val_loader, loss_fn, criterion, device, args):
    os.makedirs("checkpoints", exist_ok=True)
    n_iter = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    records = []
    test_records = []

    for epoch_counter in range(args['epochs']):

        for images, _ in tqdm(train_loader):
            images = torch.cat(images, dim=0)
            images = images.to(device)
            features = model(images)
            logits, labels = loss_fn(features, device, args)
            loss = criterion(logits, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            n_iter += 1
        
        print(f"Epoch: {epoch_counter}\tLoss: {loss.item():.4f}\t")
        records.append({"epoch": epoch_counter, "loss": loss.item()})
        if epoch_counter >= 10:
            scheduler.step()

        if epoch_counter % args["log_every_n_steps"] == 0:
            eval_res = contrastive_evaluation(model, val_loader, device, loss_fn, criterion, args)
            top1, top5 = eval_res["top1"], eval_res["top5"]
            test_loss = eval_res["loss"]
            print(f"Epoch: {epoch_counter}\tTop1 accuracy: {top1:.4f}\tTop5 accuracy: {top5:.4f}\tTest loss: {test_loss:.4f}")
            torch.save(model.state_dict(), f"checkpoints/{args['model']}_{args['dataset']}_{epoch_counter}.pt")
            test_records.append(eval_res)

    print("Training has finished.")
    torch.save(model.state_dict(), f"checkpoints/{args['model']}_{args['dataset']}_{epoch_counter}.pt")
    return records, test_records

if __name__ == "__main__":
    import pandas as pd
    from dataset import SimCLRDataset
    from model import SimCLRCNN
    on_linux = sys.platform.startswith('linux')
    args = {
        "dataset": "cifar10",
        "model": "resnet50",
        "batch_size": 1024,
        "sample_rate": 1,
        "epochs": 100,
        "n_views": 2,
        "out_dim": 128,
        "lr": 12e-4,
        "wd": 1e-6,
        "log_every_n_steps": 5,
        "n_workers": 16,
        "temperature": 0.07,
        "learning": "contrastive",
        "val_split": 0.2,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = SimCLRDataset(args["dataset"])
    build_dataloader = lambda dataset: torch.utils.data.DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=args["n_workers"],
    )

    train_dataset, val_dataset = data.get_train_val_datasets(args["n_views"], args["val_split"])
    train_loader = build_dataloader(train_dataset)
    val_loader = build_dataloader(val_dataset)
    test_dataset = data.get_test_dataset(args["n_views"])
    test_loader = build_dataloader(test_dataset)
    num_classes = data.num_classes

    model_args = {
        "backbone": args["model"],
        "out_dim": args["out_dim"] if args["learning"] == "contrastive" else num_classes,
        "mod": args["learning"] == "contrastive",
    }
    model = SimCLRCNN(**model_args).to(device)

    if on_linux:
        model = torch.compile(model)
        torch.set_float32_matmul_precision('high')

    if args["learning"] == "contrastive":
        loss_fn = info_nce_loss
        criterion = torch.nn.CrossEntropyLoss()
        train_records, test_records = contrastive_training(model, train_loader, val_loader, loss_fn, criterion, device, args)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        train_records, test_records = supervised_training(model, train_loader, val_loader, loss_fn, device, args)

    timestamp = pd.Timestamp.now().strftime("%m%d%H%M")

    df = pd.DataFrame.from_records(train_records)
    test_df = pd.DataFrame.from_records(test_records)
    df.to_csv(f"logs/{args['model']}_{args['dataset']}_{timestamp}_train.csv", index=False)
    test_df.to_csv(f"logs/{args['model']}_{args['dataset']}_{timestamp}_test.csv", index=False)
