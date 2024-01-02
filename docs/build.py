#!/usr/bin/env python3
import importlib
import os.path as osp
import subprocess
import sys


# utils
script_dir = osp.dirname(osp.realpath(__file__))


def exit_with_error(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


# check dependencies
python_packages = ['sphinx', 'sphinx_rtd_theme', 'excalibur']
for package in python_packages:
    try:
        importlib.import_module(package)
    except ModuleNotFoundError:
        exit_with_error(f"Error: required python package '{package}' not found.\n"
                         "Did you install the package with the [develop] option?")


# build documentation using Sphinx
print("\n############################################")
print("### Building Documentation using Sphinx  ###")
print("############################################")
sphinx = subprocess.Popen(['make', 'html'], cwd=script_dir)
sphinx.wait()
if sphinx.returncode != 0:
    exit_with_error("An error occured while building the Python documentation.")


# print main pages
print("\nSucessfully built the documentation:")
print(f" - Main page:  {osp.join(script_dir, 'build', 'html', 'index.html')}")
