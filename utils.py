# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch


def is_improvement(value, best, mode, epsilon):
    if mode == "min":
        return (best - value) > epsilon
    else:
        return (value - best) > epsilon


class TrainCheckpoint:
    def __init__(self, filepath, mode="min", epsilon=0.0):
        assert mode in {"min", "max"}

        self.filepath = filepath
        self.mode = mode
        self.epsilon = epsilon
        self.reinit()

    def reinit(self):
        self.best_value = float("inf") if self.mode == "min" else -float("inf")

    def step(self, value, checkpoint_dict):
        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            self.best_value = value
            torch.save(checkpoint_dict, self.filepath)


class EarlyStopping:
    def __init__(self, patience, epsilon=1e-4, mode="min"):
        assert mode in {"min", "max"}

        self.patience = patience
        self.epsilon = epsilon
        self.mode = mode
        self.reinit()

    def reinit(self):
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.wait = 0

    def step(self, value) -> bool:
        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            print(f"* metrics improved from {self.best_value:.6f} to {value:.6f}")
            self.best_value = value
            self.wait = 0
            return False
        else:
            print(f"metrics did not improve from {self.best_value:.6f}")

        self.wait += 1
        early_stop = self.wait >= self.patience
        if early_stop:
            print(f"Early stopping, no improvment in the last {self.patience} epoch")

        return early_stop


class ReduceLROnPlateau:
    def __init__(
        self,
        factor,
        patience,
        cooldown,
        mode="min",
        epsilon=1e-4,
        min_lr=1e-6,
    ):
        assert mode in {"min", "max"}

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.epsilon = epsilon
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.reinit()

    def reinit(self):
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.wait = 0
        self.cooldown_counter = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def step(self, value, optimizer_dict):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            self.best_value = value
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= self.patience:
            for optimizer in optimizer_dict.values():
                for group in optimizer.param_groups:
                    old_lr = group["lr"]
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    group["lr"] = new_lr

            self.wait = 0
            self.cooldown_counter = self.cooldown
            print(f"scheduling LR in the next epoch from {old_lr:.6f} to {new_lr:.6f}")


class TrainingCallback:
    def __init__(
        self,
        checkpoint: TrainCheckpoint = None,
        early_stop: EarlyStopping = None,
        reduce_lr: ReduceLROnPlateau = None,
    ):
        self.cp = checkpoint
        self.es = early_stop
        self.rlr = reduce_lr

    def init(self):
        if self.cp:
            self.cp.reinit()
        if self.es:
            self.es.reinit()
        if self.rlr:
            self.rlr.reinit()

    def step(self, monitor_value, model_dict=None, optimizer_dict=None) -> bool:
        # checkpoint
        if self.cp and model_dict:
            dict = {"model": model_dict}

            if optimizer_dict:
                dict["optimizer"] = optimizer_dict

            self.cp.step(monitor_value, dict)

        # lr scheduler
        if self.rlr and optimizer_dict:
            self.rlr.step(monitor_value, optimizer_dict)

        # early stop
        if self.es:
            early_stop = self.es.step(monitor_value)

            if self.rlr:
                if self.rlr.in_cooldown():
                    return False

            return early_stop

        return False


def time_formatter(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s,1)}"
