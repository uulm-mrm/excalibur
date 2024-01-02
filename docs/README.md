# Excalibur Documentation

The following instructions describe how to build the documentation pages and which dependencies are required.
In case all dependencies are available, you can also use the `build.py` script for creating the documentation automatically.

The documentation is created using Sphinx (https://www.sphinx-doc.org).
All requirements can be installed using:

```bash
bash> python3 -m pip install .[develop]
zsh>  python3 -m pip install .\[develop\]
```

To build the documentation, run the following command from the `docs` directory:

```bash
make html
```

The documentation main page is then located at `build/html/index.html`. 
