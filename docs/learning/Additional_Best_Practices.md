# Additional Best Practices

<details>
<summary><b>Use Miniconda</b></summary>

It's usually unnecessary to install full anaconda environment, miniconda should be enough (weights around 80MB).

Big advantage of conda is that it allows for installing packages without requiring certain compilers or libraries to be available in the system (since it installs precompiled binaries), so it often makes it easier to install some dependencies e.g. cudatoolkit for GPU support.

It also allows you to access your environments globally which might be more convenient than creating new local environment for every project.

Example installation:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Update conda:

```bash
conda update -n base -c defaults conda
```

Create new conda environment:

```bash
conda create -n myenv python=3.10
conda activate myenv
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System specific variables (e.g. absolute paths to datasets) should not be under version control or it will result in conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.<br>

Template contains `.env.example` file, which serves as an example. Create a new file called `.env` (this name is excluded from version control in .gitignore).
You should use it for storing environment variables like this:

```
MY_VAR=/home/user/my_system_path
```

All variables from `.env` are loaded in `train.py` automatically.

Hydra allows you to reference any env variable in `.yaml` configs like this:

```yaml
path_to_data: ${oc.env:MY_VAR}
```

</details>

<details>
<summary><b>Name metrics using '/' character</b></summary>

Depending on which logger you're using, it's often useful to define metric name with `/` character:

```python
self.log("train/loss", loss)
```

This way loggers will treat your metrics as belonging to different sections, which helps to get them organised in UI.

</details>

<details>
<summary><b>Use torchmetrics</b></summary>

Use official [torchmetrics](https://github.com/PytorchLightning/metrics) library to ensure proper calculation of metrics. This is especially important for multi-GPU training!

For example, instead of calculating accuracy by yourself, you should use the provided `Accuracy` class like this:

```python
from torchmetrics.classification.accuracy import Accuracy


class LitModel(LightningModule):
    def __init__(self)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        ...
        acc = self.train_acc(predictions, targets)
        self.log("train/acc", acc)
        ...

    def validation_step(self, batch, batch_idx):
        ...
        acc = self.val_acc(predictions, targets)
        self.log("val/acc", acc)
        ...
```

Make sure to use different metric instance for each step to ensure proper value reduction over all GPU processes.

Torchmetrics provides metrics for most use cases, like F1 score or confusion matrix. Read [documentation](https://torchmetrics.readthedocs.io/en/latest/#more-reading) for more.

</details>

<details>
<summary><b>Follow PyTorch Lightning style guide</b></summary>

The style guide is available [here](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).<br>

1. Be explicit in your init. Try to define all the relevant defaults so that the user doesn’t have to guess. Provide type hints. This way your module is reusable across projects!

   ```python
   class LitModel(LightningModule):
       def __init__(self, layer_size: int = 256, lr: float = 0.001):
   ```

2. Preserve the recommended method order.

   ```python
   class LitModel(LightningModule):

       def __init__():
           ...

       def forward():
           ...

       def training_step():
           ...

       def training_step_end():
           ...

       def on_train_epoch_end():
           ...

       def validation_step():
           ...

       def validation_step_end():
           ...

       def on_validation_epoch_end():
           ...

       def test_step():
           ...

       def test_step_end():
           ...

       def on_test_epoch_end():
           ...

       def configure_optimizers():
           ...

       def any_extra_hook():
           ...
   ```

</details>

<details>
<summary><b>Use Tmux</b></summary>

Tmux is a terminal multiplexer, which allows you to run multiple terminal sessions in a single window. It's especially useful when you want to run your training script on a remote server and you want to keep it running even after you close the ssh connection.

More about tmux can be found [here](https://hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/).
</details>

<details>
<summary><b>Specify the GPU device</b></summary>

When running your script on a server with multiple GPUs, you should specify which GPU to use. You can do this by setting the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

This will make sure that your script uses only the first GPU. If you want to use multiple GPUs, you can specify them like this:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

</details>