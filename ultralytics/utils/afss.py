# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.utils import LOGGER, LOCAL_RANK, RANK


class AFSSScheduler:
    """Anti-Forgetting Sampling Strategy (AFSS) scheduler for YOLO training.

    This scheduler partitions images into easy, moderate, and hard sets based on
    per-image precision and recall, then samples from each set according to a
    budget policy that prevents forgetting by periodically forcing long-unseen
    images back into the training batch.

    Attributes:
        num_images (int): Total number of images in the dataset.
        seed (int): Random seed for deterministic sampling.
        state (dict[int, dict]): Per-image state containing precision, recall, and last_seen_epoch.
    """

    def __init__(self, num_images: int, seed: int = 0):
        """Initialize AFSSScheduler with the given dataset size and warmup configuration.

        Args:
            num_images (int): Total number of images in the dataset.
            seed (int): Random seed for deterministic sampling.
        """
        self.num_images = num_images
        self.seed = seed
        self.state = {i: {"precision": 0.0, "recall": 0.0, "last_seen_epoch": -1} for i in range(num_images)}

    def sample_indices(self, epoch: int) -> list[int]:
        """Sample image indices for the given epoch according to AFSS policy.

        Args:
            epoch (int): Current training epoch.

        Returns:
            (list[int]): List of selected image indices.
        """
        rng = np.random.RandomState(epoch + self.seed)
        selected = []

        easy_set = []
        moderate_set = []
        hard_set = []

        for i, st in self.state.items():
            s_i = min(st["precision"], st["recall"])
            if s_i > 0.85:
                easy_set.append(i)
            elif s_i >= 0.55:
                moderate_set.append(i)
            else:
                hard_set.append(i)

        # Hard set: include all hard images
        selected.extend(hard_set)

        # Easy set
        if easy_set:
            forced_easy = [
                i for i in easy_set if (epoch - 1 - self.state[i]["last_seen_epoch"]) >= 10
            ]  # spec-defined formula
            easy_budget = round(0.02 * len(easy_set))
            forced_easy_quota = min(len(forced_easy), math.floor(0.5 * easy_budget))
            random_easy_quota = max(easy_budget - forced_easy_quota, 0)

            if easy_budget > 0:
                forced_easy_sample = []
                if forced_easy_quota > 0 and forced_easy:
                    forced_easy_sample = rng.choice(forced_easy, size=forced_easy_quota, replace=False).tolist()
                selected.extend(forced_easy_sample)

                remaining_easy = [i for i in easy_set if i not in forced_easy_sample]
                if random_easy_quota > 0 and remaining_easy:
                    random_easy_sample = rng.choice(
                        remaining_easy, size=min(random_easy_quota, len(remaining_easy)), replace=False
                    ).tolist()
                    selected.extend(random_easy_sample)

        # Moderate set
        if moderate_set:
            forced_moderate = [
                i
                for i in moderate_set
                if (epoch - 1 - self.state[i]["last_seen_epoch"]) >= 3  # spec-defined formula
            ]
            M1 = round(0.4 * len(moderate_set)) - len(forced_moderate)
            random_moderate_quota = max(min(len(moderate_set) - len(forced_moderate), M1), 0)

            selected.extend(forced_moderate)

            remaining_moderate = [i for i in moderate_set if i not in forced_moderate]
            if random_moderate_quota > 0 and remaining_moderate:
                random_moderate_sample = rng.choice(
                    remaining_moderate, size=min(random_moderate_quota, len(remaining_moderate)), replace=False
                ).tolist()
                selected.extend(random_moderate_sample)

        selected_indices = sorted(selected)
        if not selected_indices:
            LOGGER.warning(f"AFSS sampled zero images for epoch {epoch}; falling back to full dataset.")
            selected_indices = list(range(self.num_images))
        return selected_indices

    def update_last_seen(self, indices: list[int], epoch: int) -> None:
        """Update last_seen_epoch for the given indices.

        Args:
            indices (list[int]): List of image indices that were seen this epoch.
            epoch (int): Current training epoch.
        """
        for i in indices:
            self.state[i]["last_seen_epoch"] = epoch

    def update_metrics(self, image_metrics: dict[str, dict], filename_to_idx: dict[str, int]) -> None:
        """Update per-image precision and recall from validator metrics.

        Args:
            image_metrics (dict[str, dict]): Dict keyed by image filename with precision/recall values.
            filename_to_idx (dict[str, int]): Mapping from filename to dataset index.
        """
        for filename, metrics in image_metrics.items():
            idx = filename_to_idx.get(Path(filename).name)
            if idx is None:
                continue
            self.state[idx]["precision"] = float(metrics.get("precision", 0.0))
            self.state[idx]["recall"] = float(metrics.get("recall", 0.0))


def _unwrap_dataset(dataset):
    """Unwrap a dataset from any dataloader wrapper layers."""
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def afss_on_epoch_start(trainer):
    """AFSS callback: sample active indices at the start of each epoch after warmup."""
    if not hasattr(trainer, "afss_scheduler"):
        # Lazy init on first epoch
        dataset = _unwrap_dataset(trainer.train_loader.dataset)

        trainer.afss_scheduler = AFSSScheduler(len(dataset), seed=trainer.args.seed)
        trainer.afss_current_indices = list(range(len(dataset)))

        # Resume: restore scheduler state if available
        afss_path = trainer.wdir / "afss_state.pt"
        if afss_path.exists():
            state = torch.load(afss_path, weights_only=False)
            if len(state) == trainer.afss_scheduler.num_images:
                trainer.afss_scheduler.state = state
            else:
                LOGGER.warning(
                    f"AFSS resume state mismatch: expected {trainer.afss_scheduler.num_images} images, "
                    f"got {len(state)}. Starting with fresh AFSS state."
                )

    epoch = trainer.epoch
    if epoch < trainer.args.warmup_epochs:  # do not use afss during warmup
        return

    selected_indices = trainer.afss_scheduler.sample_indices(epoch)

    # DDP broadcast
    if trainer.world_size > 1:
        if RANK == 0:
            broadcast_list = [selected_indices]
        else:
            broadcast_list = [None]
        dist.broadcast_object_list(broadcast_list, src=0)
        selected_indices = broadcast_list[0]

    old_nb = trainer.nb
    if trainer.world_size > 1:
        # Rebuild loader for DDP so DistributedSampler sees new length
        batch_size = trainer.batch_size // trainer.world_size
        old_loader = trainer.train_loader
        trainer.train_loader = trainer.get_dataloader(
            trainer.data["train"],
            batch_size=batch_size,
            rank=LOCAL_RANK,
            mode="train",
            active_indices=selected_indices,
        )
        del old_loader

        new_dataset = _unwrap_dataset(trainer.train_loader.dataset)

        if trainer.args.close_mosaic and epoch >= (trainer.epochs - trainer.args.close_mosaic):
            new_dataset.close_mosaic(hyp=copy(trainer.args))
    else:
        dataset = _unwrap_dataset(trainer.train_loader.dataset)
        dataset.active_indices = selected_indices
        trainer.train_loader.reset()

    trainer.afss_current_indices = selected_indices
    trainer.nb = len(trainer.train_loader)
    # Adjust last_opt_step so optimizer stepping continues correctly when nb changes
    if old_nb != trainer.nb:
        trainer.last_opt_step -= epoch * (old_nb - trainer.nb)
    LOGGER.info(f"AFSS epoch {epoch}: training on {len(selected_indices)}/{trainer.afss_scheduler.num_images} images")


def afss_on_epoch_end(trainer):
    """AFSS callback: update last seen and refresh metrics at the end of each epoch."""
    if not hasattr(trainer, "afss_scheduler"):
        return
    epoch = trainer.epoch
    trainer.afss_scheduler.update_last_seen(trainer.afss_current_indices, epoch)
    if epoch >= trainer.args.warmup_epochs and (epoch - math.ceil(trainer.args.warmup_epochs)) % 5 == 0:
        afss_refresh_metrics(trainer)


def afss_refresh_metrics(trainer):
    """Run validation on the training set to refresh per-image precision/recall for AFSS."""
    batch_size = trainer.batch_size // max(trainer.world_size, 1)
    train_eval_loader = trainer.get_dataloader(
        trainer.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="val"
    )

    validator = trainer.get_validator().__class__(
        train_eval_loader,
        save_dir=trainer.save_dir / "afss_train_eval",
        args=copy(trainer.args),
        _callbacks=trainer.callbacks,
    )
    validator(trainer)
    del train_eval_loader

    if RANK in {-1, 0}:
        image_metrics = validator.metrics.box.image_metrics
        dataset = _unwrap_dataset(trainer.train_loader.dataset)
        filename_to_idx = {Path(f).name: i for i, f in enumerate(dataset.im_files)}
        trainer.afss_scheduler.update_metrics(image_metrics, filename_to_idx)
        LOGGER.info(f"AFSS: refreshed metrics for {len(image_metrics)} images")

    if trainer.world_size > 1:
        state_list = [trainer.afss_scheduler.state if RANK == 0 else None]
        dist.broadcast_object_list(state_list, src=0)
        trainer.afss_scheduler.state = state_list[0]


def afss_save_state(trainer):
    """Save AFSS scheduler state to a sidecar checkpoint file."""
    if hasattr(trainer, "afss_scheduler") and RANK in {-1, 0}:
        torch.save(trainer.afss_scheduler.state, trainer.wdir / "afss_state.pt")
