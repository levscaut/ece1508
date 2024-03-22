import torch
import torch.nn.functional as F

from utils import build_dataloader

'''
def info_nce_loss(features, device, args):
    labels = torch.cat(
        [torch.arange(args["batch_size"]) for i in range(args["n_views"])], dim=0
    )
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args["temperature"]
    return logits, labels
'''

def info_nce_loss(features, device, args):
    """
    Compute logits and labels for the InfoNCE loss from a batch of features,
    considering the specified device and arguments.

    Args:
        features: Tensor of shape [2*batch_size, feature_dim] where
                  2*batch_size is the total number of features, consisting of
                  batch_size pairs of positive examples. Each pair should be
                  adjacent in the tensor (i.e., 0 and 1 are a pair, 2 and 3 are
                  a pair, etc.).
        device: The device (CPU or CUDA) where tensors should be moved.
        args: Dictionary containing configurations, specifically 'temperature' for scaling the dot product in similarity calculation.

    Returns:
        logits: Tensor of shape [2*batch_size, 2*batch_size] containing the similarity scores between all pairs.
        labels: Tensor of shape [2*batch_size] containing the indices of the positive examples for each element in the batch.
    """
    temperature = args['temperature']
    features = features.to(device)

    # Normalize features
    features = F.normalize(features, p=2, dim=1)

    # Compute similarity as dot product
    anchor_dot_contrast = torch.matmul(features, features.T) / temperature

    # Generate labels
    batch_size = features.size(0) // 2
    labels = torch.arange(0, batch_size, dtype=torch.long, device=device)
    labels = torch.cat([labels, labels], dim=0)  # Duplicate for both views

    # Subtract max for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    return logits, labels

def topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def standard_evaluation(model, test_dataset, device, criterion, args):
    test_loader = build_dataloader(test_dataset, args["batch_size"], args["n_workers"])
    model.eval()
    total_loss, total_top1, total_top5, = 0.0, 0.0, 0.0
    total_num = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            top1, top5 = topk(logits, labels, topk=(1, 5))
            total_top1 += top1.item()
            total_top5 += top5.item()
            total_num += 1
    model.train()

    return {
        "loss": total_loss / total_num,
        "top1": total_top1 / total_num,
        "top5": total_top5 / total_num,
    }


def contrastive_evaluation(model, test_dataset, device, loss_fn, criterion, args):
    test_loader = build_dataloader(test_dataset, args["batch_size"], args["n_workers"])
    model.eval()
    total_loss, total_top1, total_top5 = 0.0, 0.0, 0.0
    total_num = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = torch.cat(images, dim=0).to(device)
            features = model(images)
            logits, labels = info_nce_loss(features, device, args)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            top1, top5 = topk(logits, labels, topk=(1, 5))
            total_top1 += top1.item()
            total_top5 += top5.item()
            total_num += 1
    model.train()
    return {
        "loss": total_loss / total_num,
        "top1": total_top1 / total_num,
        "top5": total_top5 / total_num,
    }
