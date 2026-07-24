import argparse
from collections.abc import Sequence
from pathlib import Path

from xxtrain.training import review


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', type=Path)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(argv)
    review(args.scenario, args.weights, args.directory)


if __name__ == '__main__':
    main()
