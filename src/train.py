import argparse
from collections.abc import Sequence
from pathlib import Path

from xxtrain.training import train


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', type=Path)
    args = parser.parse_args(argv)
    train(args.scenario)


if __name__ == '__main__':
    main()
