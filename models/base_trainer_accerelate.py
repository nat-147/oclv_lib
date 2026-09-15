import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict

import torch
import torch.nn as nn
from accelerate import Accelerator
from ignite.engine import Engine, Events
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from data.datasets import MultiObjectDataset
from evaluation.metrics.metrics_evaluator_vs_acc import MetricsEvaluator
from models.base_model import BaseModel
from models.utils import ForwardPass, TrainCheckpointHandler, infer_model_type, save_best_model, load_best_model
from utils.logging import TBLogger
from utils.utils import ExitResubmitException, SkipTrainingException

import wandb
import time

@dataclass
class BaseTrainer:
    device: str
    steps: int
    optimizer_config: DictConfig
    clip_grad_norm: Optional[float]
    checkpoint_steps: int
    logloss_steps: int
    logweights_steps: int
    logimages_steps: int
    logvalid_steps: int
    debug: bool
    resubmit_steps: Optional[int]
    resubmit_hours: Optional[float]
    working_dir: Path

    best_model: dict

    # ------------------------------------------------------------------ #
    # accelerator는 Optional — 단일 GPU / CPU 환경에서도 동작하도록         #
    # ------------------------------------------------------------------ #
    accelerator: Optional[Accelerator] = field(default=None)
    wandb_run: wandb = field(default=None)

    model: BaseModel = field(init=False)
    dataloaders: List[DataLoader] = field(init=False)
    optimizers: List[Optimizer] = field(init=False)
    trainer: Engine = field(init=False)
    evaluator: MetricsEvaluator = field(init=False)
    eval_step: ForwardPass = field(init=False)
    checkpoint_handler: TrainCheckpointHandler = field(init=False)
    lr_schedulers: List[_LRScheduler] = field(init=False)
    training_start: float = field(init=False)

    def __post_init__(self):
        self.steps = int(self.steps)

        if self.resubmit_steps is not None and self.resubmit_hours is not None:
            raise ValueError("resubmit_steps and resubmit_hours cannot both be set.")

        # accelerator가 없으면 더미 생성 (단일 GPU 호환)
        if self.accelerator is None:
            self.accelerator = Accelerator()

        self.checkpoint_handler = TrainCheckpointHandler(self.working_dir, self.device)
        self.lr_schedulers = []
        self.vparallel_mode = False
        self.vmodel = None

    # ------------------------------------------------------------------ #
    # 헬퍼                                                                 #
    # ------------------------------------------------------------------ #
    @property
    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    def _make_optimizers(self, **kwargs):
        alg = kwargs.pop("alg")
        opt_class = getattr(torch.optim, alg)
        # unwrap된 원본 모델의 파라미터를 사용해야 함
        params = list(self.accelerator.unwrap_model(self.model).parameters())
    
        self.optimizers = [opt_class(params, **kwargs)]

    def _setup_lr_scheduling(self):
        @self.trainer.on(Events.ITERATION_COMPLETED)
        def lr_scheduler_step(engine):
            logging.debug(f"Stepping {len(self.lr_schedulers)} schedulers")
            for scheduler in self.lr_schedulers:
                scheduler.step()

    @property
    def scalar_params(self) -> List[str]:
        return []

    @property
    @abstractmethod
    def loss_terms(self) -> List[str]:
        ...

    @property
    def param_groups(self) -> List[str]:
        return []

    # ------------------------------------------------------------------ #
    # train_step: backward를 accelerator.backward()로 교체                #
    # ------------------------------------------------------------------ #
    def train_step(self, engine: Engine, batch: dict) -> Tuple[dict, dict]:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        if self.accelerator.unwrap_model(self.model).name in ["monet-velocity", "sa-velocity", "tf-velocity"]:
            batch, output = self.eval_step(batch, step=engine.state.iteration)
        else:
            batch, output = self.eval_step(batch)
        self._check_shapes(batch, output)

        # ★ loss.backward() → accelerator.backward()
        self.accelerator.backward(output["loss"])

        if self.clip_grad_norm is not None:
            # mixed precision 사용 시 unscale이 필요하므로 accelerator 경유
            self.accelerator.unscale_gradients()
            torch.nn.utils.clip_grad_norm_(
                self.accelerator.unwrap_model(self.model).parameters(),
                self.clip_grad_norm,
                error_if_nonfinite=True,
            )

        for optimizer in self.optimizers:
            optimizer.step()

        return batch, output

    def _check_shapes(self, batch: dict, output: dict):
        bs = batch["image"].shape[0]
        if batch["image"].dim() == 5:
            return
        # unwrap해서 원본 모델 속성에 접근
        raw_model = self.accelerator.unwrap_model(self.model)
        if infer_model_type(raw_model.name) == "distributed":
            n_slots = 1
            repr_shape = (bs, raw_model.num_slots * raw_model.slot_size)
        else:
            n_slots = raw_model.num_slots
            repr_shape = (bs, raw_model.num_representation_slots, raw_model.slot_size)
        c = self.dataloaders[0].dataset.input_channels
        h, w = raw_model.height, raw_model.width
        assert output["loss"].dim() == 0
        assert output["mask"].shape == (bs, n_slots, 1, h, w)
        assert output["slot"].shape == (bs, n_slots, c, h, w)
        assert output["representation"].shape == repr_shape

    def setup(
        self,
        model: BaseModel,
        dataloaders: List[DataLoader],
        load_checkpoint: bool = False,
    ):
        self._post_init(model, dataloaders)
        self._setup_training(load_checkpoint)

    # ------------------------------------------------------------------ #
    # _post_init: accelerator.prepare()로 model/optimizer/dataloader 래핑  #
    # ------------------------------------------------------------------ #
    def _post_init(self, model: BaseModel, dataloaders: List[DataLoader]):
        assert model.training is True

        self.model = model
        self.dataloaders = dataloaders

        # eval_step은 unwrap된 원본 모델로 생성 (ForwardPass 내부에서 직접 forward)
        # prepare 이후엔 self.model이 DDP 래핑 모델이므로
        # ForwardPass 생성을 prepare 이후로 이동
        training_dataset = self.dataloaders[0].dataset

        tensorboard_dir = (
            self.working_dir
            / "tensorboard"
            / f"{model.name}-{training_dataset.identifier}"
        )

        # ── optimizer 먼저 생성 (prepare에 함께 넘겨야 함) ──────────────
        self._make_optimizers(**self.optimizer_config)

        # ── accelerator.prepare(): model, optimizer, dataloader 한 번에 ──
        prepared = self.accelerator.prepare(
            self.model, self.optimizers[0], *self.dataloaders
        )
        self.model = prepared[0]
        self.optimizers = [prepared[1]]
        self.dataloaders = list(prepared[2:])

        # ── prepare 이후 eval_step 생성 ─────────────────────────────────
        self.eval_step = ForwardPass(self.model, self.device)
            

        self.trainer = Engine(self.train_step)        
        if self.is_main_process:
            loss_terms = self.loss_terms
            self.logger = TBLogger(
                tensorboard_dir,
                self.trainer,
                self.accelerator.unwrap_model(self.model),
                dataloaders=self.dataloaders,
                loss_terms=loss_terms,
                scalar_params=self.scalar_params,
                event_images=Events.ITERATION_COMPLETED(every=self.logimages_steps),
                event_parameters=Events.ITERATION_COMPLETED(every=self.logweights_steps),
                event_loss=Events.ITERATION_COMPLETED(every=self.logloss_steps),
                event_stats=Events.ITERATION_COMPLETED(every=self.logloss_steps),
                param_groups=self.param_groups,
                use_wandb=True,
                wandb_run=self.wandb_run,
                num_images=4,
            )

        if len(self.dataloaders) < 2:
            raise ValueError("At least 2 dataloaders required (train and validation)")
        self.training_dataloader = self.dataloaders[0]
        self.validation_dataloader = self.dataloaders[1]

    def _setup_training(self, load_checkpoint: bool):
        self._setup_lr_scheduling()

        if load_checkpoint:
            self.checkpoint_handler.load_checkpoint(self._get_checkpoint_state())
            if self.is_main_process:
                logging.info(f"Restored checkpoint from {self.working_dir}")
            self.best_model = load_best_model(self.working_dir)
            if self.is_main_process:
                logging.info(f"Restored best model from {self.working_dir}")

        self.trainer.state.epoch_length = self.steps
        self.trainer.state.epoch = 0

        iter_start = self.trainer.state.iteration
        if self.is_main_process:
            logging.info(f"Current training iteration: {iter_start}")
        if iter_start >= self.steps:
            if self.is_main_process:
                logging.warning(
                    f"Skipping training: max steps={self.steps}, "
                    f"checkpoint={iter_start}>={self.steps}."
                )
            self.trainer.terminate()
            raise SkipTrainingException()

        loss_terms = self.loss_terms
        self.evaluator = MetricsEvaluator(
            dataloader=self.validation_dataloader,
            device=self.device,
            loss_terms=loss_terms,
            skip_background=True,
            _forward_pass=self.eval_step,
            accelerator=self.accelerator
        )

        # ── 평가 / 체크포인트 / best model은 main process에서만 저장 ─────
        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.logvalid_steps))
        def evaluate(trainer):
            self.accelerator.wait_for_everyone()
            if self.is_main_process:
                logging.info("Starting evaluation")
            self.accelerator.unwrap_model(self.model).eval()
        
            losses, metrics = self.evaluator.eval(
            self.accelerator.unwrap_model(self.model),
            steps=100
            )
            if self.is_main_process:
                self.logger.log_dict(
                    metrics=losses,
                    iteration_num=self.trainer.state.iteration,
                    group_name="validation losses",
                )
                self.logger.log_dict(
                    metrics=metrics,
                    iteration_num=self.trainer.state.iteration,
                    group_name="validation metrics",
                )
                if self.logger.use_wandb:
                    self.wandb_run.log(self.logger._wandb_buffer, step=trainer.state.iteration)
                    self.logger._wandb_buffer = {}

            self.accelerator.unwrap_model(self.model).train()
   
            if self.is_main_process:
                logging.info("Evaluation ended")
                if losses["loss"] < self.best_model["valid_loss"]:
                    state_dicts = extract_state_dicts(self._get_checkpoint_state())
                    save_best_model(self.working_dir, state_dicts)
                    self.best_model = {
                        "image_ari": metrics.get("image_ari"),
                        "video_ari": metrics.get("video_ari"),
                        "mse": metrics.get("mse"),
                        "valid_loss": losses["loss"],
                        "step": trainer.state.iteration,
                    }
                    logging.info("Best model updated")

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.checkpoint_steps))
        def save_checkpoint(engine):
            # ★ 모든 프로세스가 동기화된 후 main process에서만 저장
            self.accelerator.wait_for_everyone()
            if self.is_main_process:
                state_dicts = extract_state_dicts(self._get_checkpoint_state())
                self.checkpoint_handler.save_checkpoint(state_dicts, self.best_model)

        if self.resubmit_steps is not None:
            if self.is_main_process:
                logging.info(f"Will stop and resubmit every {self.resubmit_steps} steps")

            @self.trainer.on(Events.ITERATION_COMPLETED(every=self.resubmit_steps))
            def stop_training_resubmit_steps(engine):
                if engine.state.iteration >= self.steps:
                    if self.is_main_process:
                        logging.info("Training will terminate normally.")
                    engine.terminate()
                    return
                if self.is_main_process:
                    logging.info(
                        f"Training ended at iteration {engine.state.iteration}: "
                        f"automatic resubmission every {self.resubmit_steps} iterations."
                    )
                engine.terminate()
                raise ExitResubmitException()

        if self.resubmit_hours is not None:
            if self.is_main_process:
                logging.info(f"Will stop and resubmit every {self.resubmit_hours} hours")

            @self.trainer.on(Events.ITERATION_COMPLETED(every=self.checkpoint_steps))
            def stop_training_resubmit_hours(engine):
                diff = (time.perf_counter() - self.training_start) / 3600
                if diff < self.resubmit_hours:
                    return
                if engine.state.iteration >= self.steps:
                    if self.is_main_process:
                        logging.info("Training will terminate normally.")
                    engine.terminate()
                    return
                if self.is_main_process:
                    logging.info(
                        f"Training ended at iteration {engine.state.iteration}: "
                        f"resubmission after {self.resubmit_hours}h (now {diff:.2f}h)."
                    )
                engine.terminate()
                raise ExitResubmitException()

    def train(self):
        self.training_start = time.perf_counter()
        self.trainer.run(
            self.training_dataloader, max_epochs=1, epoch_length=self.steps
        )

    def _get_checkpoint_state(self) -> dict:
        # unwrap된 원본 모델을 저장
        state = dict(
            model=self.accelerator.unwrap_model(self.model),
            trainer=self.trainer,
        )

        state.update({f"opt_{i}": opt for i, opt in enumerate(self.optimizers)})
        if hasattr(self, "lr_schedulers"):
            state.update(
                {f"lr_scheduler_{i}": s for i, s in enumerate(self.lr_schedulers)}
            )
        logging.debug(f"State keys: {list(state.keys())}")
        return state


def extract_state_dicts(state: dict) -> dict:
    return {name: state[name].state_dict() for name in state}