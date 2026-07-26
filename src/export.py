import argparse
from collections.abc import Sequence
from pathlib import Path

from xxtrain.training import export


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', type=Path)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.scenario, args.weights)


if __name__ == '__main__':
    main()
