<center>

# Deep Learning Best Practices

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/antonibigata/minimal-lightning-hydra-template.git"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<img alt="Python Version" src="https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10-blue">

</center>


## Description

This project aims to showcase best practices and tools essential for initiating a successful deep learning project. It will cover the use of configuration files for project settings management, adopting a modular code architecture, and utilizing frameworks like Hydra for efficient configuration. The project will also focus on effective result logging and employing templates for project structuring, aiding in maintainability, scalability, and collaborative ease.

## Learning outcomes 

1. **Using Wandb to Log Training Metrics and Images**
   - Master the integration of Wandb (Weights & Biases) into your project for comprehensive logging of training metrics and images. This includes setting up Wandb, configuring it for your project, and utilizing its powerful dashboard for real-time monitoring and analysis of model performance.

2. **Using Hydra to Manage Configurations**
   - Learn to leverage Hydra for advanced configuration management in your projects. Understand how to define, organize, and override configurations dynamically, enabling flexible experimentation and streamlined management of complex projects.

3. **Using PyTorch Lightning to Train Models**
   - Gain expertise in using PyTorch Lightning to simplify the training of machine learning models. This includes setting up models, training loops, validation, testing, and leveraging PyTorch Lightning's abstractions for cleaner, more maintainable code.

4. **Einops for Easy Tensor Manipulation**
   - Acquire the skills to use Einops for intuitive and efficient tensor operations, enhancing the readability and scalability of your data manipulation code. Learn to apply Einops for reshaping, repeating, and rearranging tensors in a more understandable way.

5. **Learning to Start Training on GPU**
   - Understand how to utilize GPUs for training your models. This outcome covers the basics of GPU acceleration, including how to select and allocate GPU resources for your training jobs to improve computational efficiency.

6. **Using a Template to Start Project**
   - Familiarize yourself with starting new projects using a predefined template, specifically the [Minimal Lightning Hydra Template](https://github.com/antonibigata/minimal-lightning-hydra-template). Learn the benefits of using templates for project initialization, including predefined directory structures, configuration files, and sample code to kickstart your development process.

## Getting Familiar with the Libraries

To help you become acquainted with the key libraries used in this project, I have prepared learning notebooks that cover the basics of each. These interactive notebooks are an excellent way to get hands-on experience. You can find them in the [learning](learning) folder. The notebooks are designed to introduce you to the core concepts and functionalities of each library:

- **[Pytorch Lightning](docs/learning/Learning_about_lightning.ipynb)**: This notebook introduces Pytorch Lightning, a library that simplifies the training process of PyTorch models. It covers basic concepts like creating models, training loops, and leveraging Lightning's built-in functionalities for more efficient training.
- **[Hydra](docs/learning/Learning_about_hydra.ipynb)**: Hydra is a framework for elegantly configuring complex applications. This notebook will guide you through its configuration management capabilities, demonstrating how to streamline your project's settings and parameters.
- **[Einops](docs/learning/Learning_about_einops.ipynb)**: Einops is a library for tensor operations and manipulation. Learn how to use Einops for more readable and maintainable tensor transformations in this notebook.

For a more comprehensive understanding, I also recommend the following tutorials. They provide in-depth knowledge and are great resources for both beginners and experienced users:

- **[Pytorch Lightning Tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)**: An official guide to starting a new project with Pytorch Lightning, offering step-by-step instructions and best practices.
- **[Hydra Documentation](https://hydra.cc/docs/intro)**: The official introduction to Hydra, covering its core principles and how to integrate it into your applications.
- **[Wandb Quickstart](https://docs.wandb.ai/quickstart)**: A quickstart guide for Wandb, a tool for experiment tracking, visualization, and comparison. Learn how to integrate Wandb into your machine learning projects.
- **[Einops Basics](https://einops.rocks/1-einops-basics/)**: An introductory tutorial to Einops, focusing on the basics and fundamental concepts of the library.

By exploring these notebooks and tutorials, you will gain a solid foundation in these libraries, which are integral to the project's development.

## Starting a New Project

To main steps to start a new project are described [here](docs/learning/Starting_a_new_project.md). This notebook will guide you through the process of initializing a new project and will showcase the best practices and tools used in this project.

## Further Learning

This project is designed to provide a comprehensive understanding of best practices in deep learning, incorporating the use of PyTorch, PyTorch Lightning, Hydra, and other essential tools. However, the field of deep learning is vast and constantly evolving. To continue your learning journey, I recommend exploring the following resources:

- **[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook?tab=readme-ov-file#why-a-tuning-playbook)**: A detailed guide focused on maximizing the performance of deep learning models. It covers aspects of deep learning training such as pipeline implementation, optimization, and hyperparameter tuning.

- **[PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)**: This resource provides a set of optimizations and best practices that can accelerate the training and inference of deep learning models in PyTorch. I also recommend exploring the official [PyTorch documentation](https://pytorch.org/tutorials/), which is a rich source of tutorials, guides, and examples.

- **[Lightning Training Tricks](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)**: PyTorch Lightning implements various techniques to aid in training, making the process more efficient and smoother.

- **[Scientific Software Best Practices](https://github.com/ImperialCollegeLondon/ReCoDE_MCMCFF)**: This exemplar project showcases best practices in developing scientific software, offering insights into structured project management.

- **Further Tools**: The deep learning ecosystem includes many other tools and libraries that can enhance your projects. For instance, [Fire](https://github.com/google/python-fire) for automatically generating command line interfaces, [DeepSpeed](https://github.com/microsoft/DeepSpeed) for efficient distributed training and inference, and [Optuna](https://optuna.org/) for advanced hyperparameter optimization.


## Project Structure

The directory structure of new project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Data configs
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
│   └── inference.yaml        <- Main config for inference
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
│   ├── datamodules              <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   ├── train.py                 <- Run training
│   ├── inference.py             <- Run inference
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for creating python package
├── requirements.txt          <- File for installing python dependencies
└── README.md
```

You will find an extra folder called `docs` that contains a collection of markdown files with best practices and explanations of how to use the project.

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
