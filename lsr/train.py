from pprint import pprint
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pprint import pprint
import logging
import wandb
import os

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def train(conf: DictConfig):
    resolved_conf = OmegaConf.to_container(conf, resolve=True)
    pprint(resolved_conf)
    os.environ["WANDB_PROJECT"] = conf.wandb.setup.project
    wandb.init(
        group=conf.exp_name,
        job_type="train",
        config=resolved_conf,
        resume=conf.wandb.resume,
        settings=wandb.Settings(start_method="fork"),
    )
    logger.info(f"Working directiory: {os.getcwd()}")
    print(conf.model)
    from lsr import config

    config.config = conf
    logger.info("Resuming from checkpoint: " + str(conf.resume_from_checkpoint))
    if "global_encoder" in conf.model: 
        logger.info(
            "Global model_name_or_dir: "
            + str(conf.model.global_encoder.encoder_name_or_dir)
            + " embedding_dim: "
            + str(conf.model.global_encoder.embedding_dim)
            + " hidden_size_encoder: "
            + str(conf.model.global_encoder.hidden_size_encoder)
        )
    trainer = instantiate(conf.trainer)
    print(conf.resume_from_checkpoint)
    trainer.train(conf.resume_from_checkpoint)
    trainer.save_model()
    wandb.finish()


if __name__ == "__main__":
    train()
