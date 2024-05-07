from GTMKT.models.models import *


def get_optimizer(args, model):
    optimizer = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Optimizer: ", optimizer)
    return optimizer


def get_loss(args):
    loss = None
    if args.loss == "crossentropyloss":
        loss = torch.nn.CrossEntropyLoss()
    print("Loss: ", loss)
    return loss


def get_model(args, embedding_dim, num_clases):
    if args.model == "GTMKT":
        return GTMKT(embedding_dim, num_clases, convs=True)
    if args.model == "GraphFormerNoConvs":
        return GTMKT(embedding_dim, num_clases, convs=False)


def graph_class_num(dataset):
    class_counts = torch.zeros(dataset.num_classes, dtype=torch.long)
    for label in range(dataset.num_classes):
        class_counts[label] = torch.sum(dataset.y == label)

    print("每个类别的图的数量:")
    for label in range(dataset.num_classes):
        print(f"类别 {label}: {class_counts[label]}")
