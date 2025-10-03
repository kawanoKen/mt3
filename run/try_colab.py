#!/usr/bin/env python3
from pathlib import Path
import argparse 
import numpy as np
import librosa
import note_seq
import tensorflow as tf
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
        gin_files = [MT3_DIR/"gin/model.gin", MT3_DIR/f"gin/{model_type}.gin"]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {"inputs": self.inputs_length, "targets": self.outputs_length}
        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)
        import jax
        import numpy as _np
        from jax.sharding import Mesh as _Mesh
        # ★ data と model の2軸を持つ 2D Mesh を常に用意（model 軸の KeyError を回避）
        try:
            from jax.experimental import mesh_utils
            n = max(1, jax.device_count())
            dev_mesh = mesh_utils.create_device_mesh((n, 1))  # (data=n, model=1)
        except Exception:
            devs = jax.devices()
            n = max(1, len(devs))
            dev_mesh = _np.array(devs).reshape((n, 1))
        self.partitioner.mesh = _Mesh(dev_mesh, ('data', 'model'))  # ← 2軸を命名

        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins))
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
        return {"encoder_input_tokens": (self.batch_size, self.inputs_length),
                "decoder_input_tokens": (self.batch_size, self.outputs_length)}

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
            partitioner=self.partitioner)
        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=str(checkpoint_path), mode="specific", dtype="float32")
        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))

    def _get_predict_fn(self, train_state_axes):
        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(
                params, batch, decoder_params={"decode_rng": None})
        # return self.partitioner.partition(
        #     partial_predict_fn,
        #     in_axis_resources=(train_state_axes.params, t5x.partitioning.PartitionSpec("data",), None),
        #     out_axis_resources=t5x.partitioning.PartitionSpec("data",),
        # )
        # 単一デバイス（CPUなど）のときは pjit を使わず素の関数を返す
        if jax.device_count() == 1:
            import jax as _jax
            return _jax.jit(partial_predict_fn)
        # マルチデバイスのときだけ pjit で分割
        return self.partitioner.partition(partial_predict_fn,
                                          in_axis_resources=(train_state_axes.params,t5x.partitioning.PartitionSpec('data',), None),
                                          out_axis_resources=t5x.partitioning.PartitionSpec('data',)
                                          )

    def predict_tokens(self, batch, seed=0):
        prediction, _ = self._predict_fn(self._train_state.params, batch, jax.random.PRNGKey(seed))
        return self.vocabulary.decode_tf(prediction).numpy()

    def __call__(self, audio):
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)
        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(ds, task_feature_lengths=self.sequence_length)
        model_ds = model_ds.batch(self.batch_size)
        inferences = (tokens for batch in model_ds.as_numpy_iterator() for tokens in self.predict_tokens(batch))
        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self.postprocess(tokens, example))
        breakpoint()
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec)
        breakpoint()
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
                additional_feature_keys=["input_times"]),
            preprocessors.add_dummy_targets,
            functools.partial(preprocessors.compute_spectrograms, spectrogram_config=self.spectrogram_config),
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

if __name__ == "__main__":
    import functools
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--ckpt_dir", default="./checkpoints/mt3")
    parser.add_argument("--model", choices=["mt3", "ismir2021"], default="mt3")
    parser.add_argument("--out_midi", default="transcribed.mid")
    parser.add_argument("--out_png", default="transcribed.png")
    args = parser.parse_args()

    # 入力音声
    y, sr = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        print(f"[WARN] Resampled to {SAMPLE_RATE} Hz")

    # モデル
    model = InferenceModel(Path(args.ckpt_dir), model_type=args.model)
    est_ns = model(y)

    # MIDI 保存
    note_seq.sequence_proto_to_midi_file(est_ns, args.out_midi)
    print(f"[OK] Wrote: {args.out_midi}")

    # 可視化の保存（任意）
    try:
        import matplotlib
        matplotlib.use("Agg")  # ここが肝: GUI不要の描画に固定
        import matplotlib.pyplot as plt
        note_seq.plot_sequence(est_ns)
        plt.tight_layout(); plt.savefig(args.out_png, dpi=160); plt.close()
        print(f"[OK] Wrote: {args.out_png}")
    except Exception as e:
        print(f"[INFO] plot skipped: {e}")
