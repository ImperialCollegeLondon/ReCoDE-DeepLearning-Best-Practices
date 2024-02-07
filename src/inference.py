import pyrootutils
import os

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
if os.getenv("DATA_ROOT") is None:
    os.environ["DATA_ROOT"] = ""

import hydra
from omegaconf import DictConfig
from torchvision.transforms import transforms
from PIL import Image

from src.utils.torch_utils import load_checkpoint
from src import utils
from src.utils.utils import configure_cfg_from_checkpoint, save_summary

log = utils.get_pylogger(__name__)
FPS = 25


def inference(cfg: DictConfig):
    assert cfg.ckpt_path, "cfg.ckpt_path is required"
    assert cfg.image_path, "cfg.image_path is required"

    cfg = configure_cfg_from_checkpoint(cfg)

    # Load model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    net = hydra.utils.instantiate(cfg.model.net)

    net = load_checkpoint(
        net,
        cfg.ckpt_path,
        allow_extra_keys=True,
        extra_key="state_dict",
        replace=("net.", ""),
        map_location="cuda",
    )
    model = hydra.utils.instantiate(cfg.model).cuda()
    model.net = net.cuda().eval()

    log.info("Instantiating loggers...")
    logger = utils.instantiate_loggers(cfg.get("logger"))
    save_summary(model, cfg.paths.output_dir, logger)
    if isinstance(logger, list) and len(logger) > 0:
        logger = logger[0]

    # Process image
    pil_image = Image.open(cfg.image_path)
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28), antialias=True)])
    image = to_tensor(pil_image).cuda()

    pred = model(image.unsqueeze(0))
    label = pred.argmax(-1).cpu().item()
    class_name = None
    if model.hparams.class_names:
        class_name = model.hparams.class_names[label]
    if class_name:
        print("Class name:", class_name)
    print("Label:", label)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    inference(cfg)


if __name__ == "__main__":
    main()
