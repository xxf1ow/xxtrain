import argparse
from collections.abc import Sequence
from pathlib import Path

from xxtrain.training import export, review, train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='xxtrain')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('scenario', type=Path)

    export_parser = subparsers.add_parser('export')
    export_parser.add_argument('scenario', type=Path)
    export_parser.add_argument('--weights', type=Path, required=True)

    review_parser = subparsers.add_parser('review')
    review_parser.add_argument('scenario', type=Path)
    review_parser.add_argument('--weights', type=Path, required=True)
    review_parser.add_argument('--directory', type=Path, required=True)
    review_parser.add_argument('--unlabeled', action='store_true')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == 'train':
        train(args.scenario)
    elif args.command == 'export':
        export(args.scenario, args.weights)
    else:
        review(args.scenario, args.weights, args.directory, args.unlabeled)
