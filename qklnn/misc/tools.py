import time
import re
import os
import importlib


def str_to_int_or_float(string):
    if not isinstance(string):
        raise ValueError("Please input a string")
    try:
        result = int(string)
    except ValueError:
        result = float(string)
    return result


def first(s):
    """Return the first element from an ordered collection
    or an arbitrary element from an unordered collection.
    Raise StopIteration if the collection is empty.
    """
    return next(iter(s.items()))


def profile(x):
    """ Placeholder decorator for memory profiling """
    return x


def notify_task_done(task, starttime=None):
    msg = "{!s} done".format(task)
    if starttime != None:
        msg += " after {:.0f}s".format(time.time() - starttime)
    print(msg)


def ordered_dict_prepend(dct, key, value, dict_setitem=dict.__setitem__):
    """Put value as 0th element in OrderedDict

    By Ashwini Chaudhary
    https://stackoverflow.com/a/16664932/3613853

    """
    if hasattr(dct, "move_to_end"):
        dct[key] = value
        dct.move_to_end(key, last=False)
    else:  # Before Python3.2
        root = dct._OrderedDict__root
        first = root[1]

        if key in dct:
            link = dct._OrderedDict__map[key]
            link_prev, link_next, _ = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            link[0] = root
            link[1] = first
            root[1] = first[0] = link
        else:
            root[1] = first[0] = dct._OrderedDict__map[key] = [root, first, key]
            dict_setitem(dct, key, value)


def parse_dataset_name(store_name):
    unstab, set, gen, dim, dataset, filter_id = re.split(
        "(?:(unstable)_|)(?:(sane|test|training|)_|)(?:gen(\d+)_)(\d+)D_(.*)_filter(\d+).h5",
        store_name,
    )[1:-1]
    if filter_id is not None:
        filter_id = int(filter_id)
    gen = int(gen)
    dim = int(dim)
    if unstab == "unstable":
        unstable = True
    elif unstab is None:
        unstable = False
    else:
        raise ValueError(
            'Could not parse unstable part "{!s}" of "{!s}"'.format(unstab, store_name)
        )

    return unstable, set, gen, dim, dataset, filter_id


def dump_package_versions(modules=None, log_func=print):
    if modules is None:
        modules = ["xarray", "netCDF4", "pandas", "tables", "dask"]
        modules = ["numpy", "pandas", "tables", "xarray", "netCDF4", "dask"]
    try:
        ModuleNotFoundError
    except NameError:
        ModuleNotFoundError = ImportError

    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            log_func("{!s} not found".format(module_name))
        else:
            log_func("{!s} is version {!s}".format(module_name, module.__version__))

    import qlknn

    repo_path = os.path.dirname(os.path.dirname(qlknn.__file__))
    try:
        from git import Repo
    except ImportError:
        # Git not found, falling back to hacky method
        head_file = os.path.join(repo_path, ".git", "HEAD")
        if os.path.isfile(head_file):
            with open(head_file) as file:
                ref = file.readlines()
            if len(ref) != 1:
                log_func("qlknn version unknown; .git/HEAD contains unexpected information")
            ref = ref[0]
            ref = ref.split(":")[1].strip()
            ref_file = os.path.join(repo_path, ".git", ref)
            with open(ref_file) as file:
                commit_sha = file.readlines()
            if len(commit_sha) != 1:
                log_func(
                    "qlknn version unknown; {!s} contains unexpected information".format(ref_file)
                )
            else:
                commit_sha = commit_sha[0].strip()
                log_func("qlknn is version {!s}. Dirtiness unknown".format(commit_sha[:8]))

        else:
            log_func("qlknn version unknown; could not find .git/HEAD at {!s}".format(head_file))
    else:
        try:
            repo = Repo(repo_path)
        except:
            log_func(
                "qlknn version unknown; could not find git repository in {!s}".format(repo_path)
            )
        else:
            commit_sha = str(repo.head.commit)
            if repo.is_dirty():
                log_func("qlknn is version {!s}".format(commit_sha[:8] + "-dirty"))
            else:
                log_func("qlknn is version {!s}".format(commit_sha[:8]))
