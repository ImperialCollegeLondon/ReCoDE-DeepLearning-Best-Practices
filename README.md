<div align="center">

# Deep Learning best practices

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This project aims to showcase best practices and tools essential for initiating a successful deep learning project. It will cover the use of configuration files for project settings management, adopting a modular code architecture, and utilizing frameworks like Hydra for efficient configuration. The project will also focus on effective result logging and employing templates for project structuring, aiding in maintainability, scalability, and collaborative ease.

## Learning outcomes (temporary)

 - Using wandb to log training metrics and Images
 - Using hydra to manage configurations
 - Using pytorch lightning to train models
 - Einops for easy tensor manipulation
 - Learning to start training on GPU (CUDA_VISIBLE_DEVICES)
 - Using a template to start project (https://github.com/antonibigata/minimal-lightning-hydra-template)

## Steps (temporary)
I think the best approch would be to create own github project based on instructions of this one.

 1. Read about the template and how to use it for new project.
 2. Create a new project based on the template.
 3. Read about hydra and how to use it for configuration management.
 4. Read about pytorch lightning and how to use it for training.
 5. Start by creating a datamodule and dataset file with corresponding config.
 6. Test it.
 7. Create model + configuration.
 8. Test it.
 (8.5 Read about einops and reimplement part of forward computation)
 9. Create trainer + configuration.
 10. Try a debug run.
 11. Learn how to use wandb for logging.
 12. Start a training run.
 13. Get results on wandb.
 14. Improve model.
 15. Start a new training run with new model via command line.
 16. Use model for inference on custom data.
 17. Use project as a python package.
 18. Give ressources for further best practices (Compiling, optimizing training loop, Fire package, etc)

## Getting Familiar with the Libraries

To help you become acquainted with the key libraries used in this project, I have prepared learning notebooks that cover the basics of each. These interactive notebooks are an excellent way to get hands-on experience. You can find them in the [learning](learning) folder. The notebooks are designed to introduce you to the core concepts and functionalities of each library:

- **[Pytorch Lightning](learning/Learning_about_lightning.ipynb)**: This notebook introduces Pytorch Lightning, a library that simplifies the training process of PyTorch models. It covers basic concepts like creating models, training loops, and leveraging Lightning's built-in functionalities for more efficient training.
- **[Hydra](learning/Learning_about_hydra.ipynb)**: Hydra is a framework for elegantly configuring complex applications. This notebook will guide you through its configuration management capabilities, demonstrating how to streamline your project's settings and parameters.
- **[Einops](learning/Learning_about_einops.ipynb)**: Einops is a library for tensor operations and manipulation. Learn how to use Einops for more readable and maintainable tensor transformations in this notebook.

For a more comprehensive understanding, I also recommend the following tutorials. They provide in-depth knowledge and are great resources for both beginners and experienced users:

- **[Pytorch Lightning Tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)**: An official guide to starting a new project with Pytorch Lightning, offering step-by-step instructions and best practices.
- **[Hydra Documentation](https://hydra.cc/docs/intro)**: The official introduction to Hydra, covering its core principles and how to integrate it into your applications.
- **[Wandb Quickstart](https://docs.wandb.ai/quickstart)**: A quickstart guide for Wandb, a tool for experiment tracking, visualization, and comparison. Learn how to integrate Wandb into your machine learning projects.
- **[Einops Basics](https://einops.rocks/1-einops-basics/)**: An introductory tutorial to Einops, focusing on the basics and fundamental concepts of the library.

By exploring these notebooks and tutorials, you will gain a solid foundation in these libraries, which are integral to the project's development.

## Steps When Starting a New Project

**1. Create a New Project Based on the Template**:
Start by using the template available at [Minimal Lightning Hydra Template](https://github.com/antonibigata/minimal-lightning-hydra-template). To create a new project from this template:
- Navigate to the GitHub page of the template.
- Click on the "Use this template" button, located in the top right corner.
- Follow the on-screen instructions to create a new repository based on this template.

Alternatively, if you prefer to clone the repository:
```bash
git clone https://github.com/antonibigata/minimal-lightning-hydra-template.git your-project-name
cd your-project-name
rm -rf .git
```
This will clone the repository and remove its git history, allowing you to start a fresh project.

**2. Create a Datamodule and Dataset File with Corresponding Configuration**:
Data manipulation code should be organized in `src/datamodules`. Key components include:
- **Datamodule**: Responsible for downloading data, and splitting it into training, validation, and testing sets.
- **Dataset**: Handles data loading and applying transformations.

Define these components in `datamodule.py` and `components/dataset.py`, respectively. Configure the datamodule in `configs/datamodule/datamodule.yaml`.

For efficient experimentation with multiple datasets, use a `default.yaml` file in the `configs` folder to define common parameters like `batch_size` and `num_workers`. Create separate configuration files for each dataset by inheriting from the default configuration. This setup allows you to switch datasets easily by modifying the datamodule configuration.

As an example, we will use the FashionMNIST dataset. I suggest writing your own datamodule and dataset files, but you can also copy the ones from this project. It is also good practice to write a small test to ensure that the datamodule and dataset are functioning as expected. Examples of these tests can be found at the end of both files.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Project Structure

The directory structure of new project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

You will find an extra folder called `learning` that contains a collection of markdown files with best practices and explanations of how to use the project.

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
