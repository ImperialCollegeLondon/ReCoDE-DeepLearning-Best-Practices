# Instal Repository as a Package (Optional)

Creating a Python package involves organizing your code into a structured format that can be easily distributed and installed. Utilizing `pyproject.toml` is a modern approach that specifies build system requirements for Python projects. Here's a small tutorial on how to transform the given repository into a Python package using `pyproject.toml`.

This is a minimal example of how to create a Python package. You can find more details at the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) or using this [template](https://github.com/ImperialCollegeLondon/pip-tools-template).

## Step 1: Organize Your Code

First, ensure your project has a suitable structure. A typical package structure looks like this:

```
your_project_name/
│
├── src/
│   └── your_package_name/
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
│
├── tests/
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_module2.py
│
├── pyproject.toml
└── README.md
```

- `your_project_name/` is the root directory.
- `src/` contains all your source files. It's a good practice to keep your package code inside a `src/` directory to avoid import issues.
- `your_package_name/` is the directory that will be the actual Python package.
- `tests/` contains your unit tests.
- `pyproject.toml` will contain your package metadata and build requirements.

As you see the current structure needs to be slightly modified to fit the package structure. The only change is to move everything from the `src` folder to `src/your_package_name`. This is the only change needed to make the project a package.

## Step 2: Create the `pyproject.toml` File

The `pyproject.toml` file is a configuration file to define your package metadata and build system requirements. Explanation of the file can be found [here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

## Step 3: Add an `__init__.py` File

Inside your package directory (`src/your_package_name/`), make sure there's an `__init__.py` file. This file can be empty, but its presence indicates to Python that this directory should be treated as a package.

## Step 4: Build Your Package

With your code properly organized and your `pyproject.toml` in place, you can now build your package. First, ensure you have the required tools installed:

```bash
pip install setuptools wheel
```

Navigate to your project root directory and run:

```bash
python -m build
```

This command generates distribution files in the `dist/` directory.

## Step 5: Publish Your Package (Optional)

If you want to upload your package to the Python Package Index, the first thing you’ll need to do is register an account on TestPyPI, which is a separate instance of the package index intended for testing and experimentation. It’s great for things like this tutorial where we don’t necessarily want to upload to the real index. To register an account, go to https://test.pypi.org/account/register/ and complete the steps on that page. You will also need to verify your email address before you’re able to upload any packages. For more details, see Using [TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/).

To securely upload your project, you’ll need a PyPI [API token](https://test.pypi.org/help/#apitoken). Create one at https://test.pypi.org/manage/account/#api-tokens, setting the “Scope” to “Entire account”. Don’t close the page until you have copied and saved the token — you won’t see that token again.

Now that you are registered, you can use [twine](https://packaging.python.org/en/latest/key_projects/#twine) to upload the distribution packages. You’ll need to install Twine:

```bash
pip install twine
```

Then, upload your package by running:

```bash
twine upload dist/*
```

You'll need a PyPI account and to follow the prompts to authenticate.

## Step 6: Install Your Package

It's useful for development and testing purposes. Navigate to the root directory of your project (where `setup.py` or `pyproject.toml` is located) and run:

```bash
pip install .
```

Or if you're actively developing the package and want changes in the package to be immediately reflected without needing to reinstall, use:

```bash
pip install -e .
```

The `-e` flag installs the package in "editable" mode.

## Additional Considerations

- **Dependencies**: If your package has dependencies listed in `pyproject.toml`, they will be automatically installed by `pip` during the installation process.
- **Virtual Environment**: It's a good practice to install your package in a virtual environment to avoid conflicts with system-wide packages. You can create a virtual environment using `python -m venv env` and activate it with `source env/bin/activate` (on Unix/macOS) or `env\Scripts\activate` (on Windows).
- **Uninstalling**: You can uninstall your package at any time with `pip uninstall your_package_name`.

By following these methods, you can easily install your Python package locally for development, testing, or personal use without the need to publish it to PyPI.