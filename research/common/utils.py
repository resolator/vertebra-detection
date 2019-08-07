# -*- coding: utf-8 -*-
import os


def check_or_create_dir(path, critical=False):
    """Check the given path for existence and try to create it if not exists.

    Parameters
    ----------
    path : str
        Path for check.
    critical : bool
        If it's impossible to create a given dir exit with error

    """
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileNotFoundError:
            err_desc = 'can\'t find or create the path: ' + path
            if critical:
                print('\nERROR: ' + err_desc)
                exit(1)
            else:
                print('\nWARNING: ' + err_desc)
