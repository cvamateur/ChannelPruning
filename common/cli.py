import argparse

from common.path import DATA_ROOT


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default=DATA_ROOT, help="Data root directory")

    parser.add_argument("--download", action="store_true", help="Download MNIST dataset")
    parser.add_argument("--log-freq", type=int, default=2, help="Logging frequency")
    parser.add_argument("--eval-epoch", type=int, default=1, help="Evaluate start and this epoch")
    parser.add_argument("--eval-freq", type=int, default=1, help="Evaluation frequency")

    parser.add_argument("--factor", type=float, default=1.0, help="Factor of model width")

    parser.add_argument("--num-workers", type=int, default=0, help="Number workers for dataset")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N")
    parser.add_argument("--num-epochs", type=int, default=5, metavar="N")
    parser.add_argument("--lr", type=float, default=1e-2, metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="Weight decay for L2 regularization")

    parser.add_argument("--pruning_ratio", type=float, default=0.8, help="pruning ratio")

    return parser


