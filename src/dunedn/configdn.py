# This file is part of DUNEdn by M. Rossi
import os
from pathlib import Path


def get_dunedn_path():
    """
    Loads DUNEdn package source directory path from environment variable.

    Returns
    -------
        - Path, the path to DUNEdn package source directory
    """
    root = Path(os.environ.get("DUNEDN_PATH"))
    if root is not None:
        return root
    else:
        error_msg = f"""
Please, make the environment variable DUNEDN_PATH point to the DUNEdn repository root directory"""
        raise RuntimeError(error_msg)


def get_dunedn_search_path():
    """
    Retrieves the list of directories to look for the configuration card.
    Loads DUNEDN_SEARCH_PATH from environment variable (a colon separated
    list of folders).
    The first item is automatically set to the current directory.
    The last item is fixed to the configcards folder in the DUNEdn package.

    Set this variable with:
        `export DUNEDN_SEARCH_PATH=<new path>:$DUNEDN_SEARCH_PATH`

    Returns
    -------
        - list, of Path objects from DUNEDN_SEARCH_PATH
    """
    # get directories from colon separated list
    env_var = os.environ.get("DUNEDN_SEARCH_PATH")
    search_path = [] if env_var is None else env_var.split(":")

    
    # prepend current directory
    search_path.insert(0, ".")

    # append the configcards directory
    search_path.append(get_dunedn_path() / "configcards")

    # remove duplicates
    search_path = list(dict.fromkeys(search_path))

    # turn elements into Path objects
    return list(map(Path, search_path))
    