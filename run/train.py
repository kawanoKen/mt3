#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal trainer for MT3 on top of T5X/SeqIO.

- Uses SeqIO Tasks/Mixtures registered by `mt3.tasks`
- Calls t5x.train.train(...) directly (no external launcher needed)
- Fine-tuning from an existing checkpoint is supported via --restore

Tested conceptually against T5X train API (see citations in the write-up).
"""

from pathlib import Path
import argparse
import os
import numpy as np
import gin
import jax
import jax.numpy as jnp
import tensorflow as tf
import seqio
import t5x
import tasks_local
from t5x import train as t5x_train

from jax.sharding import Mesh

# MT3 modules
from mt3 import models, network, spectrograms, vocabularies
import mt3 as mt3_pkg  # for locating gin files


from clu import periodic_actions
from tqdm import tqdm

class ProgressBar(periodic_actions.PeriodicAction):
    def __init__(self, total_steps: int, every_steps: int = 1):
        super().__init__(every_steps=every_steps)
        self.total_steps = int(total_steps)
        self.pbar = None
        self._last_step = 0

    def _maybe_init(self):
        if self.pbar is None:
            # マルチホスト時は1プロセスだけ表示
            if jax.process_index() == 0:
                self.pbar = tqdm(total=self.total_steps, dynamic_ncols=True)
            else:
                # 他プロセスでは何もしない
                self.pbar = False  # sentinel

    def _apply(self, step, t=None):
        # PeriodicAction が呼ぶ実体。step は 0 始まり。
        self._maybe_init()
        if self.pbar is False:
            return
        # tqdm は増分更新が安全
        inc = int(step) - self._last_step
        if inc > 0:
            self.pbar.update(inc)
            self._last_step = int(step)

    def close(self):
        if self.pbar not in (None, False):
            self.pbar.close()

def build_partitioner():
    """Create a 2D (data, model) mesh even on single device to avoid axis errors."""
    part = t5x.partitioning.PjitPartitioner(num_partitions=1)
    try:
        from jax.experimental import mesh_utils
        n = max(1, jax.device_count())
        dev_mesh = mesh_utils.create_device_mesh((n, 1))  # (data=n, model=1)
    except Exception:
        devs = jax.devices()
        n = max(1, len(devs))
        dev_mesh = np.array(devs).reshape((n, 1))
    part.mesh = Mesh(dev_mesh, ("data", "model"))
    return part


def build_model(model_type: str, num_velocity_bins: int):
    """Construct MT3 model + output features (vocabulary)."""
    if model_type == "mt3":
        inputs_length_default = 256
    elif model_type == "ismir2021":
        inputs_length_default = 512
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    # Spectrogram config (controls input depth)
    spec_cfg = spectrograms.SpectrogramConfig()

    # Build codec/vocabulary for event tokens (targets)
    codec = vocabularies.build_codec(
        vocab_config=vocabularies.VocabularyConfig(
            num_velocity_bins=num_velocity_bins
        )
    )
    vocabulary = vocabularies.vocabulary_from_codec(codec)

    # Output features: continuous inputs (spectrogram), discrete targets (event tokens)
    output_features = {
        "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
        "targets": seqio.Feature(vocabulary=vocabulary),
    }

    # Parse MT3's model gin (keeps model parity with original code)
    mt3_dir = Path(mt3_pkg.__file__).resolve().parent
    gin_files = [mt3_dir / "gin/model.gin", mt3_dir / f"gin/{model_type}.gin"]
    with gin.unlock_config():
        gin.parse_config_files_and_bindings(
            [str(p) for p in gin_files],
            [
                "from __gin__ import dynamic_registration",
                "from mt3 import vocabularies",
                "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
                f"vocabularies.VocabularyConfig.num_velocity_bins={num_velocity_bins}",
            ],
            finalize_config=False,
        )

    model_config = gin.get_configurable(network.T5Config)()
    module = network.Transformer(config=model_config)

    # T5X model wrapper
    model = models.ContinuousInputsEncoderDecoderModel(
        module=module,
        input_vocabulary=output_features["inputs"].vocabulary,   # None for continuous
        output_vocabulary=output_features["targets"].vocabulary, # event vocab
        optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
        input_depth=spectrograms.input_depth(spec_cfg),
    )
    return model, output_features, spec_cfg, inputs_length_default


def summarize_gin_to_tb(model_dir: str, writer, step: int):
    # """Minimal summarize_config_fn compatible with t5x.train.train(...)"""
    # if jax.process_index() == 0:
    #     writer.write_texts({"gin/operative_config": gin.operative_config_str()}, step)
    #     writer.flush()
     return

def constant_lr_fn(lr: float):
    # JAX/TF 両対応のスカラーを返す
    return lambda step: jnp.array(lr, dtype=jnp.float32)  # ← TF ではなく JAX を返す


def parse_args():
    p = argparse.ArgumentParser(description="Train / fine-tune MT3 with T5X/SeqIO")

    # What to train on
    p.add_argument(
        "--mixture_or_task",
        type=str,
        help="SeqIO Mixture/Task name registered by mt3.tasks（例：--list_tasks で候補を確認）",
    )
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")

    # Model
    p.add_argument("--model", choices=["mt3", "ismir2021"], default="mt3")
    p.add_argument("--num_velocity_bins", type=int, default=None,
                   help="mt3: 1 / ismir2021: 127（未指定なら自動）")

    # Sequence lengths
    p.add_argument("--inputs_length", type=int, default=None,
                   help="未指定ならモデル種別のデフォルト（mt3:256 / ismir:512）")
    p.add_argument("--targets_length", type=int, default=1024)

    # Batch / steps
    p.add_argument("--batch_size", type=int, default= 1)
    p.add_argument("--total_steps", type=int, default=100000)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--eval_period", type=int, default=1000)
    p.add_argument("--stats_period", type=int, default=None)

    # Checkpoints
    p.add_argument("--ckpt_dir", type=str,
                   help="学習出力（チェックポイント・サマリー）保存先")
    p.add_argument("--ckpt_period", type=int, default=1000)
    p.add_argument("--keep", type=int, default=3)
    p.add_argument("--restore", type=str, default="",
                   help="fine-tune 元チェックポイント（ディレクトリ or 具体パス）")

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="",
                   help="TFDS / データキャッシュのルート（必要に応じて）")
    p.add_argument("--list_tasks", action="store_true",
                   help="mt3.tasks を import し、利用可能な Task / Mixture 名を列挙して終了")
    p.add_argument("--force_cpu", action="store_true",
                   help="TensorFlow/GPU を無効化（CPUでの検証用）")

    args = p.parse_args()

    # --list_tasks のときは必須チェックをスキップ
    if not args.list_tasks:
        if args.mixture_or_task is None or args.ckpt_dir is None:
            p.error("--mixture_or_task と --ckpt_dir は必須です")

    return args



def main():
    args = parse_args()

    # Optional: force TF to CPU (JAX 側はデバイスに合わせて自動)
    if args.force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            print("[INFO] TensorFlow pinned to CPU.")
        except Exception as e:
            print("[INFO] TF GPU disable skipped:", e)

    # Ensure TFDS dir if specified (mt3.tasks の一部は TFDS を参照)
    if args.data_dir:
        os.environ["TFDS_DATA_DIR"] = args.data_dir

    # Importing registers tasks/mixtures into SeqIO
    from mt3 import tasks as mt3_tasks  # noqa: F401

    if args.list_tasks:
        names = sorted(seqio.TaskRegistry.names())
        print("\n== Registered SeqIO Task / Mixture names ==")
        for n in names:
            print(n)
        return

    # Model + vocabulary
    if args.model == "mt3":
        nv = 1 if args.num_velocity_bins is None else args.num_velocity_bins
    else:
        nv = 127 if args.num_velocity_bins is None else args.num_velocity_bins

    model, output_features, spec_cfg, inputs_len_default = build_model(args.model, nv)

    # Sequence lengths
    inputs_len = args.inputs_length or inputs_len_default
    targets_len = args.targets_length
    seq_len = {"inputs": inputs_len, "targets": targets_len}

    # Partitioner / mesh
    partitioner = build_partitioner()

    # 学習用データセット
    train_cfg = t5x.utils.DatasetConfig(
        mixture_or_task_name=args.mixture_or_task,
        task_feature_lengths=seq_len,
        split=args.train_split,
        batch_size=args.batch_size,
         #shuffle=True,
        shuffle=False,   # デバッグ用にシャッフルしない
        seed=args.seed,
    )

    # # 評価用
    # train_eval_cfg = t5x.utils.DatasetConfig(
    #     mixture_or_task_name=args.mixture_or_task,
    #     task_feature_lengths=seq_len,
    #     split=args.eval_split,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     seed=args.seed,
    # ) if args.eval_steps and args.eval_steps > 0 else None
    train_eval_cfg = None


# 置き換え後（OK）
    save_cfg = t5x.utils.SaveCheckpointConfig(
        period=args.ckpt_period,
        keep=args.keep,
    )
    restore_cfg = (
        t5x.utils.RestoreCheckpointConfig(
            path=args.restore, mode="specific", dtype="float32"
        )
        if args.restore
        else None
    )
    ckpt_cfg = t5x.utils.CheckpointConfig(save=save_cfg, restore=restore_cfg)

    lr_value = getattr(args, "lr", 1e-3)     # CLIを追加していないなら既定1e-3
    lr_fn = constant_lr_fn(lr_value)
    from functools import partial
    TrainerWithDefaults = partial(
        t5x.trainer.Trainer,
        learning_rate_fn=lr_fn,
        num_microbatches=None,        # マイクロバッチ分割しないなら None
    )

    # Kick training
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print("[INFO] Starting training...")
    abs_model_dir = str(Path(args.ckpt_dir).expanduser().resolve())
    t5x_train.train(
        model=model,
        train_dataset_cfg=train_cfg,
        train_eval_dataset_cfg=train_eval_cfg,
        infer_eval_dataset_cfg=None,
        checkpoint_cfg=ckpt_cfg,
        partitioner=partitioner,
        trainer_cls=TrainerWithDefaults,  # ← ここを差し替え
        model_dir=abs_model_dir,
        total_steps=args.total_steps,
        eval_steps=args.eval_steps,
        eval_period=args.eval_period,
        stats_period=args.stats_period,
        random_seed=args.seed,
        use_hardware_rng=False,
        summarize_config_fn=summarize_gin_to_tb,
        inference_evaluator_cls=seqio.Evaluator,
        get_dataset_fn=t5x.utils.get_dataset,
        concurrent_metrics=False,
        actions={"TRAIN": [ProgressBar(args.total_steps, every_steps=1)]},
        train_eval_get_dataset_fn=t5x.utils.get_training_eval_datasets,
        run_eval_before_training=False,
    )

    print("[OK] Training finished.")


if __name__ == "__main__":
    main()
