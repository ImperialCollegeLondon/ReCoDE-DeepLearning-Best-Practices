<div align="center">

# Deep Learning best practices

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/antonibigata/minimal-lightning-hydra-template.git"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
![Python Version](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10-blue)

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

### **1. Create a New Project Based on the Template**:
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

### **2. Create a Datamodule and Dataset File with Corresponding Configuration**:
Data manipulation code should be organized in `src/datamodules`. Key components include:
- **Datamodule**: Responsible for downloading data, and splitting it into training, validation, and testing sets.
- **Dataset**: Handles data loading and applying transformations.

Define these components in `datamodule.py` and `components/dataset.py`, respectively. Configure the datamodule in `configs/datamodule/datamodule.yaml`.

For efficient experimentation with multiple datasets, use a `default.yaml` file in the `configs` folder to define common parameters like `batch_size` and `num_workers`. Create separate configuration files for each dataset by inheriting from the default configuration. This setup allows you to switch datasets easily by modifying the datamodule configuration.

As an example, we will use the FashionMNIST dataset. I suggest writing your own datamodule and dataset files, but you can also copy the ones from this project. It is also good practice to write a small test to ensure that the datamodule and dataset are functioning as expected. Examples of these tests can be found at the end of both files.

### **3. Create Networks with Corresponding Configuration**:
Your main model will rely on networks, which are essentially the your building blocks, such as layers or a set of layers that perform specific functions (like encoders and decoders in autoencoders). Define these networks in `src/models/components/nets`. Networks are versatile and can be reused across different models. For instance, an encoder network designed for one model can potentially be used in another model with similar requirements.

Network configurations should be placed in `configs/model/net`. This maintains organization and clarity, especially when dealing with multiple networks.

As a practical exercise, start by creating a simple model and network. Even if it's basic, this exercise will help you understand the process of building and integrating these components. Use the examples provided in this project as a reference or solution. To ensure your model and networks are set up correctly, it's advisable to write unit tests. These tests verify the functionality of each component. You can find examples of such tests at the end of the respective files in this project.

By following these steps, you'll gain hands-on experience in setting up and configuring models and networks, which are crucial for developing effective machine learning solutions.

### **4. Create Model and Configuration**:
In this step, you will define your main model within the `src/models` directory. This model should be a PyTorch Lightning module, which means it inherits from `LightningModule`. The model encapsulates your machine learning algorithm, including methods for the forward pass, loss calculation, and the optimizer step. It's also where you'll define the main training, validation, and testing loops.

For configuration, similar to the datamodule, set up a default configuration in `configs/model/default.yaml`. This file should contain settings common to all models. Then, for specific models, create separate configuration files within the same directory. This approach allows for flexibility and ease in switching between different model configurations.

To get started, you're encouraged to write your own model, but do take the time to examine the example model included in this project. It serves as a practical reference to understand the standard structure and components of a PyTorch Lightning module. Additionally, refer to the PyTorch Lightning documentation, particularly the LightningModule section at [LightningModule Documentation](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html). This resource is invaluable for understanding the different methods in a LightningModule and how they interact in a real-world scenario. By studying these examples and the documentation, you will gain a deeper insight into the efficient implementation and functionalities of your machine learning model.

### **5. Modify Training Configuration**:
At this point, you have assembled all the essential components needed to commence the training of your model. Central to this process is the PyTorch Lightning `Trainer`, which orchestrates the training workflow. The Trainer manages crucial aspects of the training process, such as:

- Number of training epochs.
- Logging of training progress and metrics.
- Checkpointing, which involves saving the model at specific intervals or under certain conditions.

The configuration settings for the Trainer are located in `configs/trainer`. This directory contains a set of default configurations tailored to different training scenarios. These default settings can be conveniently overridden via command-line arguments. For example, to modify the number of training epochs, you can execute a command like:

```bash
python src/train.py trainer.max_epochs=20
```

Let's dive into the `configs/train.yaml` file, which serves as the master configuration encapsulating all modules of your project:

1. **datamodule**: Specifies the datamodule configuration, as outlined in section 2. This includes details about data preprocessing, batching, and split between training, validation, and test sets.

2. **model**: Contains the model configuration as discussed earlier. It defines the structure and parameters of your machine learning model.

3. **callbacks**: Manages callbacks used during training. The initial setup typically includes logging callbacks, but you can activate additional callbacks like early stopping or model checkpointing by modifying `configs/callbacks/defaults.yaml`.

4. **logger**: Handles the configuration for logging. More details on this will be covered in the next section. For now, it can be left as `null`.

5. **trainer**: Defines trainer-specific configurations, such as the number of epochs, the number of devices (GPUs) to use, and other training-related settings.

6. **paths**: This section outlines various paths used by the project, including paths to the data, logs, and checkpoints. You can stick with the default settings unless you have specific requirements.

By configuring these components in the `configs/train.yaml` file, you create a cohesive and flexible training environment. This setup allows you to easily adjust parameters and components to suit different experiments and project needs.

### **6. Start the Training**:
Now that you have set up all the necessary components, you are ready to initiate the training of your model. Begin by executing the following command:

```bash
python src/train.py
```

This command launches the main training loop by running the `train.py` script. Within `train.py`, several key actions are performed:

- Initialization of the datamodule, which prepares your data for the training process.
- Setting up the model, tailored to your specific machine learning task.
- Configuring and starting the PyTorch Lightning Trainer, which manages the training process.

In addition to orchestrating the training, `train.py` is also responsible for handling logging. More details about logging will be discussed in the subsequent section.

The above command initiates training using the default configuration settings. However, Hydra's flexible design allows you to easily override configuration parameters directly from the command line. For instance, if you want to switch the network architecture within your model, you can do so with a command like:

```bash
python src/train.py model/net=conv_net
```

Note: To change an entire configuration file (as opposed to a single parameter), use `/` instead of `.` to separate configuration names.

[OPTIONAL GPU TRAINING]
If you have access to a GPU, you can significantly accelerate the training process by specifying the `trainer=gpu` argument:

```bash
python src/train.py trainer=gpu
```

This command instructs the Trainer to utilize available GPU resources, harnessing their computational power for more efficient training.

### **7. Enhanced Experiment Tracking and Visualization**:

Logging training progress and metrics is crucial for monitoring a model's performance throughout the training process. This functionality is configured within the `logger` section of `configs/train.yaml`. By default, the `logger` is set to `null`, indicating that no logging will occur. However, enabling logging is straightforward and highly recommended for a more insightful training experience.


[Wandb (Weights & Biases)](https://wandb.ai) is a widely used platform for experiment tracking, visualization, and analysis. It offers a comprehensive suite of tools to log metrics, hyperparameters, outputs, and much more. To incorporate Wandb into your project:

- 1. **Installation**: First, ensure that the Wandb library is installed. If not, you can install it using pip:

```bash
pip install wandb
```

- 2. **Create an Account**: Visit the [Wandb website](https://wandb.ai) to sign up for an account. Signing up is free and allows you to track and visualize your experiments in one place.

- 3. **Authenticate**: Obtain your Wandb API key from your account settings. Then, authenticate your local environment by running:

```bash
wandb login
```

This command prompts you to enter your API key, securely linking your experiments with your Wandb account.

- 4. **Enable Wandb Logging**: To start logging to Wandb, modify the `logger` section in `configs/train.yaml` by setting it to `wandb`:

```yaml
logger: wandb
```

Alternatively, you can directly activate Wandb logging via the command line when you start your training:

```bash
python src/train.py logger=wandb
```

**What Can You Log?**

Wandb's flexibility allows you to log a wide array of data, from training and validation metrics to rich media such as images, videos, and even audio clips. This capability is invaluable for in-depth monitoring and analysis of your training process, offering insights that can guide model improvement.

For detailed instructions on logging specific data types and leveraging Wandb's full potential, refer to the [Wandb Documentation](https://docs.wandb.ai/guides/track/log). Here, you'll find a treasure trove of guides and examples to enhance your logging strategy.

**Logging Images and Confusion Matrices**

In addition to the standard training and validation metrics, this tutorial also covers how to log images and confusion matrices using Wandb. Logging images can provide visual feedback on the model's performance, such as the accuracy of image classifications or the quality of generated images. Confusion matrices, on the other hand, offer a detailed view of the model's prediction accuracy across different classes, helping identify areas where the model may be struggling. Both of these logging capabilities are powerful tools for diagnosing model behavior and guiding improvements. Detailed instructions on how to implement these logging features are included, allowing you to gain deeper insights into your model's performance and make data-driven decisions to enhance its accuracy and effectiveness.

### **8. Evaluate the model**:

After completing the training phase, it's crucial to assess the model's performance using a separate test set. This step is vital for gauging the model's generalization abilities—essentially, its capacity to make accurate predictions on new, unseen data. This unbiased evaluation helps in understanding how the model would perform in real-world scenarios.

**Initiating the Model Evaluation**

To start the evaluation process, use the following command:

```bash
python src/eval.py ckpt_path=PATH_TO_CHECKPOINT
```

Here, `PATH_TO_CHECKPOINT` should be replaced with the actual path to your saved model checkpoint, for example, `logs/train/runs/2024-02-05_10-41-42`. This path points to the specific model folder checkpoint you intend to evaluate. The `eval.py` script is designed to load this checkpoint and then carry out the evaluation on the designated test dataset.

**Note on Dataset Usage**:

For illustrative purposes, the evaluation script in this project is configured to run on the FashionMNIST dataset's validation set. However, for a thorough evaluation of your model, especially when working with your own datasets, it is strongly recommended to perform this assessment on a dedicated test set. The test set should consist of data that has not been used during the training or validation phases to ensure an impartial evaluation of the model's performance.

**Locating the Checkpoint File**:

The checkpoint file is typically saved in a directory structured by the training date and time, for example, `logs/train/runs/YYYY-MM-DD_HH-MM-SS`. If you're unsure about the checkpoint path, refer to the training logs or the directory where your training outputs are saved. This can help you identify the correct checkpoint file for evaluation.

**Best Practices for Evaluation**:

- Ensure that the test set is properly prepared and represents the data distribution expected in real-world applications.
- Consider evaluating the model on multiple checkpoints to identify the best-performing model over different stages of training.
- Use a comprehensive set of metrics for evaluation, tailored to your specific problem, to get a holistic view of the model's performance.

### **9. Model Inference on Custom Data**:

After your model has been rigorously trained and its performance thoroughly evaluated, the next step is to deploy it for practical use — a phase known as inference. Inference is the application of your trained model to make predictions or decisions based on new, unseen data. This step is critical for realizing the model's value in real-world applications, from classifying images to making recommendations.

**Performing Inference on Custom Data**

To apply your model to custom data, utilize the `inference.py` script with the following command structure:

```bash
python src/inference.py ckpt_path=PATH_TO_CHECKPOINT image_path=PATH_TO_IMAGE
```

- `PATH_TO_CHECKPOINT` should be substituted with the actual path to your trained model's checkpoint. This file encapsulates the learned weights that your model will use to make predictions.
- `PATH_TO_IMAGE` needs to be replaced with the path to the specific image file you wish to analyze with your model.

In this particular project, the inference script is designed to work with the FashionMNIST dataset so the image needs to be a FashionMNIST image (i.e 28x28 black and white image). However, you can easily modify the script to work with your own custom data. To facilitate immediate testing of the inference capabilities of your model, the repository includes two sample images from the FashionMNIST dataset. These images are located in logs/data_check. This inclusion allows you to quickly test the model's inference process without the need for external data. To use these for inference testing, simply replace PATH_TO_IMAGE in the inference command with the path to one of these sample images.

The script executes two primary actions:
1. **Model Loading**: It begins by loading your model from the specified checkpoint. This step reconstitutes your model with all its learned parameters, readying it for prediction.
2. **Prediction**: Next, the script processes the provided image, applying the model to generate predictions based on the learned patterns during training.

**Practical Tips**:

- If you're working with batches of images or data points rather than a single instance, consider modifying the `inference.py` script to handle batch processing for more efficiency.
- Explore different model checkpoints to observe any variations in prediction outcomes. This can help identify the most stable and reliable version of your model for deployment.
- Keep in mind the context of your application. The nature of the data and the specific requirements of your use case should guide how you approach inference, from selecting the right model to choosing the appropriate data for prediction.


## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install requirements
pip install -r requirements.txt
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
