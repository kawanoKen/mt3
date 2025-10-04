# tasks_local.py
# -*- coding: utf-8 -*-
import functools
from pathlib import Path
import seqio
from mt3 import datasets, spectrograms, vocabularies, preprocessors
from mt3.tasks import add_transcription_task_to_registry

# ── ここをあなたの保存先に合わせてください ──
TRAIN_PATH = "/data/maestro-v3.0.0_ns_wav_train.tfrecord-00004-of-00025"
EVAL_PATH  = "/data/maestro-v3.0.0_ns_wav_train.tfrecord-00005-of-00025"

# 1) ローカルの TFRecord を指す DatasetConfig を定義
LocalCfg = datasets.DatasetConfig(
    name="maestro_local_ns_wav",
    paths={
        "train": TRAIN_PATH,         # 学習は 00004 のみ
        "eval_train": EVAL_PATH,     # 評価は 00005 のみ
    },
    # “audio が波形サンプル”のスキーマを使う（MUSICNET_EM と同じ features）
    features=datasets.MUSICNET_EM_CONFIG.features,
    train_split="train",
    train_eval_split="eval_train",
    infer_eval_splits=[],
    track_specs=None,
)

# 2) ties + vb1 (=1 velocity bin) で MT3 標準設定
SPECTRO_CONFIG = spectrograms.SpectrogramConfig()
VOCAB_VB1 = vocabularies.VocabularyConfig(num_velocity_bins=1)

add_transcription_task_to_registry(
    dataset_config=LocalCfg,
    spectrogram_config=SPECTRO_CONFIG,
    vocab_config=VOCAB_VB1,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_transcription_example,
        audio_is_samples=True,   # ← “ns_wav” は生波形を格納している想定
        id_feature_key="id"),
    onsets_only=False,
    include_ties=True,
)
