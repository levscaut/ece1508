import torch
import sys
import os
from dataset import SimCLRDataset
from model import SimCLRCNN
from train import supervised_training, standard_evaluation


def finetune(model: SimCLRCNN, dataset: SimCLRDataset, device, args):
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(args["ckpt_path"])), f"finetune_{args['dataset']}_ratio_{args['ft_ratio']}")
    num_classes = dataset.num_classes
    criterion = torch.nn.CrossEntropyLoss()
    ckpt_path = args['ckpt_path']
    model.train()
    ckpt_weights = torch.load(ckpt_path)
    ckpt_weights = {k.replace("_orig_mod.", ""): v for k, v in ckpt_weights.items()}
    model.load_state_dict(ckpt_weights)
    model.finetune(num_classes)
    model = model.to(device)
    train_dataset = dataset.get_train_dataset(1, args["ft_ratio"])
    test_dataset = dataset.get_test_dataset(1)
    
    train_records, _ = supervised_training(
        model, train_dataset, None, criterion, device, args, artifacts_dir, silent=False
    )
    test_records = standard_evaluation(
        model, test_dataset, device, criterion, args
    )

    return train_records, test_records

if __name__ == "__main__":

    torch.manual_seed(4090)
    on_linux = sys.platform.startswith("linux")
    args = {
        "dataset": "cifar10",
        "model": "resnet50",
        "batch_size": 512,
        "sample_rate": 1,
        "epochs": 100,
        "n_views": 2,
        "out_dim": 256,
        "lr": 3e-4,
        "wd": 1e-6,
        "log_every_n_steps": 5,
        "n_workers": 16,
        "temperature": 0.07,
        "learning": "contrastive",
        "val_split": 0,
        "ft_ratio": 1,
        "ckpt_path": "/home/levscaut/ece1508/artifacts/resnet50_cifar10_500epochs/checkpoints/final.pt",
        "ckpt": "",
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = SimCLRDataset(args["dataset"])
    num_classes = data.num_classes
    model_args = {
        "backbone": args["model"],
        "out_dim": args["out_dim"]
        if args["learning"] == "contrastive"
        else num_classes,
        "mod": args["learning"] == "contrastive",
    }

    model = SimCLRCNN(**model_args).to(device)

    train_records, test_records = finetune(model, data, device, args)

    print(train_records)
    print(test_records)