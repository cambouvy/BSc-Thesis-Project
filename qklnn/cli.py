""" Initialization functions for the main QLKNN CLI entrypoint"""
# pylint: disable=import-outside-toplevel
import sys
from argparse import Namespace, ArgumentParser, Action

from IPython import embed  # pylint: disable=unused-import # noqa: F401


def main(parser=None):
    """ Initialize the main QLKNN CLI entrypoint"""
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument("--verbosity", "-v", default=0, action="count")

    # See https://docs.python.org/dev/library/argparse.html#sub-commands
    subparsers: Action = parser.add_subparsers(dest="subparser_name")
    assert isinstance(subparsers, Action)

    quick_parser: ArgumentParser = subparsers.add_parser(
        "quickslicer",
        description="Valid subcommands",
        help="additional help",
        aliases=["quickslicer"],
    )
    assert isinstance(quick_parser, ArgumentParser)

    # Split off in subcommand.
    # argparse will check if it is an allowed name, but to get it
    # mid-initialize, we need to grab the CLI arguments manually
    parsed_args: Namespace = parser.parse_args([sys.argv[1]])

    # Do subcommand-specific initialization of parsers
    if parsed_args.subparser_name == "quickslicer":
        from qlknn.plots.quickslicer import initialize_argument_parser as quick_init

        # pylint: disable=unused-variable
        quickslicer: ArgumentParser = quick_init(quick_parser)  # noqa: F841
        args = parser.parse_args()
    elif parsed_args.subparser_name == "clustering":
        raise NotImplementedError

    # Call the actual subcommand routine.
    # func should be set by the subcommand initialization routines themselves
    args.func(args)


if __name__ == "__main__":
    main()
