import logging
import platform
from pathlib import Path

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import hydra
from omegaconf import OmegaConf
import wandb

from accelerate import Accelerator

from data.datasets import make_dataloaders, hydra_to_dataset_args
from models.base_model import BaseModel
from models.base_trainer_accerelate import BaseTrainer
from models.nn_utils import summary_num_params
from utils.logging import logging_wrapper, setup_logging
from utils.paths import CONFIG, ROOT
from utils.utils import (
    SkipTrainingException,
    add_uuid,
    available_cuda_device_names,
    get_cli_overrides,
    omegaconf_equal,
    set_all_seeds,
)

import torch
from velocity_extraction.utils import get_dataset_dataloader
from velocity_extraction.data import ContinuousShiftingSpritesloops


@hydra.main(config_path=str(CONFIG), config_name="train_object_discovery")
def main(config):
    # ------------------------------------------------------------------
    # 1. Accelerator 초기화
    # ------------------------------------------------------------------
    torch.cuda.empty_cache()
    
    accelerator = Accelerator()

    curr_dir = Path.cwd()
    curr_dir.mkdir(parents=True, exist_ok=True)

    # main process 에서만 로그 파일 생성
    if accelerator.is_main_process:
        setup_logging(log_fname="train.log")

    cli_overrides = get_cli_overrides()

    if accelerator.is_main_process:
        logging.info(f"Running on node '{platform.node()}'")
        logging.info(f"Available cuda devices: {available_cuda_device_names()}")
        logging.info(f"DDP: {accelerator.num_processes} process(es), "
                     f"device={accelerator.device}, "
                     f"mixed_precision={accelerator.mixed_precision}")

    # ------------------------------------------------------------------
    # 2. Debug 모드 오버라이드
    # ------------------------------------------------------------------
    if config.debug:
        debug_overrides = {
            "batch_size": 2,
            "trainer.steps": 2,
            "data_sizes": [1000, 8, 8],
            "trainer.logweights_steps": 1,
            "trainer.logimages_steps": 1,
            "trainer.logloss_steps": 1,
            "trainer.checkpoint_steps": 2,
            "trainer.logvalid_steps": 2,
        }
        cli_conflicts = []
        for name, value in debug_overrides.items():
            if name in cli_overrides:
                cli_conflicts.append(name)
            hierarchy = name.split(".")
            current = config
            for key in hierarchy[:-1]:
                current = current[key]
            current[hierarchy[-1]] = value
        if cli_conflicts and accelerator.is_main_process:
            logging.warning(
                "The following arguments were specified from command line but were "
                f"overridden because the debug flag is true: {cli_conflicts}"
            )

    assert len(config.data_sizes) == 3, "Need a train/validation/test split."

    if accelerator.is_main_process:
        logging.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    add_uuid(config)

    train_config_path = curr_dir / "train_config.yaml"
    load_checkpoint = False

    if train_config_path.exists():
        if not config.allow_resume:
            raise FileExistsError(
                f"Previous run found in '{curr_dir}' but flag 'allow_resume' is False"
            )
        with open(train_config_path) as configfile:
            prev_config = OmegaConf.load(configfile)
        config.uuid = prev_config.uuid
        ignore_list = [
            "curr_dir", "allow_resume", "trainer.steps",
            "trainer.checkpoint_path", "trainer.logweights_steps",
            "trainer.logimages_steps", "trainer.logloss_steps",
            "trainer.resubmit_steps", "trainer.resubmit_hours",
            "device", "num_workers",
        ]
        if not omegaconf_equal(config, prev_config, ignore=ignore_list):
            raise RuntimeError(
                f"Attempting to resume training from '{curr_dir}' but the configs do not match"
            )
        load_checkpoint = True

    # ------------------------------------------------------------------
    # 3. wandb 초기화 (main process 에서만)
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        wandb_run = wandb.init(
            project="object-discovery",
            name=curr_dir.name,
            config=OmegaConf.to_container(config, resolve=True),
            dir=str(curr_dir),

            resume="allow" if config.allow_resume else None,
            id=config.uuid if config.allow_resume else None,
        )
    else:
        wandb_run = None

    # ------------------------------------------------------------------
    # 4. 시드 고정 (accelerate 제공 set_seed 사용)
    # ------------------------------------------------------------------
    set_all_seeds(config.seed)

    if accelerator.is_main_process:
        with open(train_config_path, "w") as f:
            OmegaConf.save(config, f)

    # ------------------------------------------------------------------
    # 5. 모델 생성
    #    device 배치는 accelerator.prepare() 에 위임하므로
    #    .to(config.device) 제거
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        logging.info("Creating model")

    model: BaseModel = hydra.utils.instantiate(config.model)

    if accelerator.is_main_process:
        logging.info(f"Model summary:\n{model}")
        summary_string, num_params = summary_num_params(model, max_depth=4)
        logging.info(f"Model parameters summary:\n{summary_string}")

    # ------------------------------------------------------------------
    # 6. 데이터로더 생성
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        logging.info("Creating data loaders")

    if config.dataset.name == 'contshiftingsprites':
        dataconfig = dict(
            dataset_class=ContinuousShiftingSpritesloops,
            dataset_args=hydra_to_dataset_args(config),
            dataloader_args=dict(
                batch_size=config.batch_size // accelerator.num_processes,
                pin_memory=True,
                num_workers=config.num_workers // accelerator.num_processes,
                persistent_workers=config.persistent_workers,
                prefetch_factor=config.prefetch_factor,
            )
        )
        _, _, train_loader, test_loader = get_dataset_dataloader(dataconfig)
        dataloaders = [train_loader, test_loader]
    else:
        dataloaders = make_dataloaders(
            dataset_config=config.dataset,
            data_sizes=config.data_sizes[:2],
            batch_size=config.batch_size // accelerator.num_processes,
            num_workers=config.num_workers // accelerator.num_processes,
            pin_memory=True,
            #persistent_workers=config.persistent_workers,
            #prefetch_factor=config.prefetch_factor,
        )

    # ------------------------------------------------------------------
    # 7. Trainer 생성
    #    device 인자를 accelerator.device 로 전달
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        logging.info("Creating trainer")

    trainer: BaseTrainer = hydra.utils.instantiate(
        config.trainer,
        device=accelerator.device,   # ← accelerator가 할당한 device 사용
        debug=config.debug,
        working_dir=curr_dir,
        best_model={"ari": 0.0, "mse": 1.0, "valid_loss": 1e6, "step": 0},
        wandb_run=wandb_run,
        accelerator=accelerator,     # ← trainer 내부에서 prepare() 호출용
    )

    # ------------------------------------------------------------------
    # 8. accelerator.prepare() — model, optimizer, scheduler, dataloader
    #    를 DDP / mixed-precision 용으로 래핑
    #
    #    trainer.setup() 내부에서 optimizer/scheduler 를 생성한다면
    #    아래 패턴 대신 trainer 내부로 prepare() 를 이동하는 것을 권장.
    #    여기서는 trainer가 accelerator를 받아서 직접 처리한다고 가정.
    # ------------------------------------------------------------------
    try:
        trainer.setup(model, dataloaders, load_checkpoint)
    except SkipTrainingException:
        accelerator.end_training()
        return

    if accelerator.is_main_process:
        logging.info("Training starts")

    torch.cuda.empty_cache()
    trainer.train()

    if accelerator.is_main_process:
        logging.info("Training completed")
        wandb.finish()
        accelerator.end_training()


if __name__ == "__main__":
    main()