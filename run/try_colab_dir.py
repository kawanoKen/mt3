#!/usr/bin/env python3
from pathlib import Path
import argparse
import functools
import sys
import traceback
import numpy as np
import librosa
import note_seq
import tensorflow as tf

# --- TF を CPU 固定（GPU 環境でも安全に動くように） ---
try:
    tf.config.set_visible_devices([], 'GPU')
    print("[INFO] TensorFlow is pinned to CPU.")
except Exception as e:
    print("[INFO] TF GPU disable skipped:", e)

import gin, jax, seqio, t5x
import t5.data.preprocessors

# mt3 関連
from mt3 import metrics_utils, models, network, note_sequences, preprocessors, spectrograms, vocabularies
import mt3 as mt3_pkg

# ---- Colab/GA を無効化（呼び出しは残っていてもOK） ----
def load_gtag(): pass
def log_event(*args, **kwargs): pass
# --------------------------------------------------------

SAMPLE_RATE = 16000
SF2_PATH = "SGM-v2.01-Sal-Guit-Bass-V1.3.sf2"  # 使わないなら未使用でOK

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".aifc"}

def is_audio_file(path: Path) -> bool:
    return path.is_file() and (path.suffix.lower() in AUDIO_EXTS)

def list_audio_files(root: Path, recursive: bool = True) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if is_audio_file(root) else []
    # directory
    if recursive:
        return [p for p in root.rglob("*") if is_audio_file(p)]
    else:
        return [p for p in root.glob("*") if is_audio_file(p)]

def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def as_output_path(
    in_audio: Path,
    out_spec: Path,
    new_suffix: str,
    root_in: Path | None = None
) -> Path:
    """
    out_spec がファイルならそのまま。
    out_spec がディレクトリなら、in_audio の相対パス（root_in からの相対があればそれを維持、なければファイル名のみ）
    に基づいて出力先ファイルパスを作る。
    """
    if out_spec.suffix:  # ファイル指定
        return out_spec
    # ディレクトリ指定
    if root_in and root_in.is_dir():
        try:
            rel = in_audio.relative_to(root_in)
            rel = rel.with_suffix(new_suffix)
            return out_spec.joinpath(rel)
        except ValueError:
            # relative_to に失敗（root_in外）→ 名前だけ使う
            return out_spec.joinpath(in_audio.stem + new_suffix)
    else:
        return out_spec.joinpath(in_audio.stem + new_suffix)

class InferenceModel(object):
    def __init__(self, checkpoint_path, model_type="mt3"):
        if model_type == "ismir2021":
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == "mt3":
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        # gin ファイルはパッケージ内から
        MT3_DIR = Path(mt3_pkg.__file__).resolve().parent
        gin_files = [MT3_DIR / "gin/model.gin", MT3_DIR / f"gin/{model_type}.gin"]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {"inputs": self.inputs_length, "targets": self.outputs_length}
        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        import jax
        import numpy as _np
        from jax.sharding import Mesh as _Mesh
        # data と model の2軸 Mesh を常に用意（model 軸 KeyError 回避）
        try:
            from jax.experimental import mesh_utils
            n = max(1, jax.device_count())
            dev_mesh = mesh_utils.create_device_mesh((n, 1))  # (data=n, model=1)
        except Exception:
            devs = jax.devices()
            n = max(1, len(devs))
            dev_mesh = _np.array(devs).reshape((n, 1))
        self.partitioner.mesh = _Mesh(dev_mesh, ('data', 'model'))

        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins)
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        self._parse_gin(gin_files)
        self.model = self._load_model()
        self.restore_from_checkpoint(checkpoint_path)

    @property
    def input_shapes(self):
        return {
            "encoder_input_tokens": (self.batch_size, self.inputs_length),
            "decoder_input_tokens": (self.batch_size, self.outputs_length),
        }

    def _parse_gin(self, gin_files):
        gin_bindings = [
            "from __gin__ import dynamic_registration",
            "from mt3 import vocabularies",
            "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
            "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(gin_files, gin_bindings, finalize_config=False)

    def _load_model(self):
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config),
        )

    def restore_from_checkpoint(self, checkpoint_path):
        train_state_initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner,
        )
        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=str(checkpoint_path), mode="specific", dtype="float32"
        )
        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0)
        )

    def _get_predict_fn(self, train_state_axes):
        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(
                params, batch, decoder_params={"decode_rng": None}
            )
        if jax.device_count() == 1:
            import jax as _jax
            return _jax.jit(partial_predict_fn)
        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(train_state_axes.params, t5x.partitioning.PartitionSpec("data",), None),
            out_axis_resources=t5x.partitioning.PartitionSpec("data",),
        )

    def predict_tokens(self, batch, seed=0):
        prediction, _ = self._predict_fn(self._train_state.params, batch, jax.random.PRNGKey(seed))
        return self.vocabulary.decode_tf(prediction).numpy()

    def __call__(self, audio: np.ndarray):
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)
        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length
        )
        model_ds = model_ds.batch(self.batch_size)
        inferences = (tokens for batch in model_ds.as_numpy_iterator() for tokens in self.predict_tokens(batch))
        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self.postprocess(tokens, example))
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec
        )
        return result["est_ns"]

    def audio_to_dataset(self, audio):
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors({"inputs": frames, "input_times": frame_times})

    def _audio_to_frames(self, audio):
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key="inputs",
                additional_feature_keys=["input_times"],
            ),
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms, spectrogram_config=self.spectrogram_config
            ),
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def postprocess(self, tokens, example):
        tokens = self._trim_eos(tokens)
        start_time = example["input_times"][0]
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {"est_tokens": tokens, "start_time": start_time, "raw_inputs": []}

    @staticmethod
    def _trim_eos(tokens):
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return tokens

def transcribe_one_audio(model: InferenceModel, audio_path: Path) -> note_seq.protobuf.music_pb2.NoteSequence:
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        print(f"[WARN] Resampled to {SAMPLE_RATE} Hz for {audio_path.name}")
    est_ns = model(y)
    return est_ns

def save_outputs(est_ns, out_midi_path: Path, out_png_path: Path | None):
    ensure_parent_dir(out_midi_path)
    note_seq.sequence_proto_to_midi_file(est_ns, str(out_midi_path))
    print(f"[OK] Wrote MIDI: {out_midi_path}")
    if out_png_path is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ensure_parent_dir(out_png_path)
            note_seq.plot_sequence(est_ns)
            plt.tight_layout()
            plt.savefig(str(out_png_path), dpi=160)
            plt.close()
            print(f"[OK] Wrote PNG : {out_png_path}")
        except Exception as e:
            print(f"[INFO] plot skipped for {out_png_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="MT3-based transcription. Supports file or folder for inputs and outputs."
    )
    parser.add_argument("--audio", required=True,
                        help="入力音声ファイル or フォルダ（拡張子: wav, mp3, flac, ogg, m4a, aac, wma, aiff など）")
    parser.add_argument("--ckpt_dir", default="./checkpoints/mt3")
    parser.add_argument("--model", choices=["mt3", "ismir2021"], default="mt3")
    parser.add_argument("--out_midi", default="transcribed.mid",
                        help="出力MIDIのファイル or フォルダ")
    parser.add_argument("--out_png", default=None,
                        help="出力PNG（ピアノロール可視化）のファイル or フォルダ。未指定なら保存しない")
    parser.add_argument("--recursive", action="store_true",
                        help="--audio がフォルダのとき、再帰的に探索する")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存ファイルがあっても上書きする")
    args = parser.parse_args()

    in_path = Path(args.audio).expanduser().resolve()
    midi_spec = Path(args.out_midi).expanduser().resolve()
    png_spec = Path(args.out_png).expanduser().resolve() if args.out_png else None

    # 入力の解決（単一ファイル or フォルダ）
    inputs = list_audio_files(in_path, recursive=args.recursive)
    if not inputs:
        if in_path.is_file():
            print(f"[ERR] Not an audio file: {in_path}", file=sys.stderr)
        else:
            print(f"[ERR] No audio files found under: {in_path}", file=sys.stderr)
        sys.exit(1)

    # モデルの準備（1回だけロードして使い回し）
    model = InferenceModel(Path(args.ckpt_dir), model_type=args.model)

    # フォルダ出力時の相対基準
    root_in_for_rel = in_path if in_path.is_dir() else in_path.parent

    total = len(inputs)
    print(f"[INFO] Found {total} audio file(s). Start transcribing...")

    num_ok, num_skip, num_err = 0, 0, 0
    for idx, a_path in enumerate(sorted(inputs), start=1):
        try:
            # 出力先の決定
            midi_out = as_output_path(
                in_audio=a_path, out_spec=midi_spec, new_suffix=".mid", root_in=root_in_for_rel
            )
            png_out = None
            if png_spec is not None:
                png_out = as_output_path(
                    in_audio=a_path, out_spec=png_spec, new_suffix=".png", root_in=root_in_for_rel
                )

            # 既存チェック
            if not args.overwrite and midi_out.exists() and (png_out is None or png_out.exists()):
                print(f"[SKIP] {a_path} -> already exists (use --overwrite to force)")
                num_skip += 1
                continue

            # 変換
            print(f"[{idx}/{total}] Transcribing: {a_path}")
            est_ns = transcribe_one_audio(model, a_path)
            save_outputs(est_ns, midi_out, png_out)
            num_ok += 1
        except Exception as e:
            print(f"[ERR] Failed: {a_path} :: {e}")
            traceback.print_exc()
            num_err += 1

    print(f"[DONE] OK={num_ok}, SKIP={num_skip}, ERR={num_err}")

if __name__ == "__main__":
    main()
