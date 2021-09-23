#!/bin/sh
# Script to set up clean Docker Python container path
# Not needed in CI
# Inspired by https://github.com/gcc-mirror/gcc/blob/master/install-sh
# Very verbose and explicit to learn shell scripting properly ;)

scriptversion=$(git describe); # UTC

# Convenient characters for shell scripting
tab='	'
nl='
'
IFS=" $tab$nl"

# This always works in bash shell
# TODO: Check portability
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ $DIR = "" ]; then
    echo Could not find rundir; exit 1
fi

# defaults
usage="\
Usage: $DIR/$0 [OPTION]...

Options:
     --help     display this help and exit.
     --version  display version info and exit.
"

verbosity=0

while test $# -ne 0; do
  case $1 in
    --help) echo "$usage"; exit $?;;
    --version) echo "$0 $scriptversion"; exit $?;;
    --verbose) verbosity=1;;
    --) shift
        break;;
    -*) echo "$0: invalid option: $1" >&2
        exit 1;;
    *)  break;;
  esac
  shift
done

# Check if this script is ran directly by the interpreter (e.g. when sourcing.) like
# . ./script.sh (sh)
# source script.sh (bash)
# or ran in a subshell, like
# ./script.sh
if [ "$_" != "$0" ]; then
    if [ $verbosity -ge 1 ]; then
        echo "Script is being sourced"
    fi
else
    if [ $verbosity -ge 1 ]; then
        echo "Script running as script"
    fi
fi

# Now process CLI arguments and do what the user asked
if test $# -eq 0; then
    if [ $verbosity -ge 1 ]; then
        echo exporting PYTHONTOOLS_VENV_PATH
        echo ROOT=$(realpath $DIR/../..)
    fi
    # export PYTHONTOOLS_VENV_PATH=/builds/Karel-van-de-Plassche/QLKNN-develop/venv/qlknn
    # In CI
    export PYTHONTOOLS_VENV_PATH=$(realpath $ROOT/venv/qlknn)
    exit 0
else
    echo Positional arguments given, not allowed
    echo "$usage"
    exit 1
fi

echo "$usage"; exit $?
