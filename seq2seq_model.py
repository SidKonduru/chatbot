"""TensorFlow 2 / Keras seq2seq chatbot model."""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_utils import prepare_decoder_inputs


class Seq2SeqModel:
    """Encoder-decoder LSTM seq2seq model with greedy inference."""

    def __init__(
        self,
        enc_max_len,
        dec_max_len,
        vocab_size,
        emb_dim=512,
        num_layers=2,
        dropout=0.3,
        learning_rate=1e-3,
        ckpt_dir="weights",
    ):
        self.enc_max_len = enc_max_len
        self.dec_max_len = dec_max_len
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.ckpt_dir = ckpt_dir

        self.train_model = self._build_train_model()
        self.encoder_model, self.decoder_model = self._build_inference_models()

    def _build_encoder(self, enc_inputs):
        enc_emb = layers.Embedding(
            self.vocab_size, self.emb_dim, mask_zero=True, name="enc_embedding"
        )(enc_inputs)

        x = enc_emb
        states = []
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            return_state = i == self.num_layers - 1
            lstm = layers.LSTM(
                self.emb_dim,
                return_sequences=return_sequences,
                return_state=return_state,
                dropout=self.dropout,
                name=f"enc_lstm_{i}",
            )
            if return_state:
                x, state_h, state_c = lstm(x)
                states = [state_h, state_c]
            else:
                x = lstm(x)

        return states

    def _build_train_model(self):
        enc_inputs = layers.Input(shape=(self.enc_max_len,), name="encoder_inputs")
        dec_inputs = layers.Input(shape=(self.dec_max_len,), name="decoder_inputs")

        enc_states = self._build_encoder(enc_inputs)

        dec_emb = layers.Embedding(
            self.vocab_size, self.emb_dim, mask_zero=True, name="dec_embedding"
        )(dec_inputs)

        dec_outputs, _, _ = layers.LSTM(
            self.emb_dim,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout,
            name="dec_lstm",
        )(dec_emb, initial_state=enc_states)

        dec_logits = layers.Dense(self.vocab_size, name="dec_dense")(dec_outputs)

        model = keras.Model([enc_inputs, dec_inputs], dec_logits)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _build_inference_models(self):
        enc_inputs = layers.Input(shape=(self.enc_max_len,), name="inf_enc_inputs")
        enc_emb = self.train_model.get_layer("enc_embedding")(enc_inputs)

        x = enc_emb
        states = []
        for i in range(self.num_layers):
            lstm_layer = self.train_model.get_layer(f"enc_lstm_{i}")
            if i == self.num_layers - 1:
                _, state_h, state_c = lstm_layer(x)
                states = [state_h, state_c]
            else:
                x = lstm_layer(x)

        encoder_model = keras.Model(enc_inputs, states)

        dec_state_inputs = [
            layers.Input(shape=(self.emb_dim,), name="dec_state_h"),
            layers.Input(shape=(self.emb_dim,), name="dec_state_c"),
        ]
        dec_input = layers.Input(shape=(1,), name="dec_input_token")

        dec_emb = self.train_model.get_layer("dec_embedding")(dec_input)
        dec_outputs, state_h, state_c = self.train_model.get_layer("dec_lstm")(
            dec_emb, initial_state=dec_state_inputs
        )
        dec_logits = self.train_model.get_layer("dec_dense")(dec_outputs)
        decoder_model = keras.Model(
            [dec_input] + dec_state_inputs,
            [dec_logits, state_h, state_c],
        )

        return encoder_model, decoder_model

    def train(
        self,
        train_x,
        train_y,
        valid_x,
        valid_y,
        batch_size=64,
        epochs=50,
        patience=5,
    ):
        """Train with early stopping and checkpointing."""
        os.makedirs(self.ckpt_dir, exist_ok=True)

        train_dec_in = prepare_decoder_inputs(train_y)
        valid_dec_in = prepare_decoder_inputs(valid_y)

        best_path = os.path.join(self.ckpt_dir, "best_model.keras")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                best_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        history = self.train_model.fit(
            [train_x, train_dec_in],
            train_y,
            validation_data=([valid_x, valid_dec_in], valid_y),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        self.save()
        return history

    def predict(self, question_indices):
        """Greedy-decode a single encoded question."""
        question = np.array(question_indices, dtype=np.int32).reshape(1, -1)
        states = self.encoder_model.predict(question, verbose=0)

        target_seq = np.zeros((1, 1), dtype=np.int32)
        output_indices = []

        for _ in range(self.dec_max_len):
            dec_outputs, h, c = self.decoder_model.predict(
                [target_seq] + states, verbose=0
            )
            sampled_index = int(np.argmax(dec_outputs[0, -1]))
            if sampled_index == 0:
                break
            output_indices.append(sampled_index)
            target_seq = np.array([[sampled_index]], dtype=np.int32)
            states = [h, c]

        return np.array(output_indices, dtype=np.int32)

    def save(self):
        """Persist model weights and hyperparameters."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        model_path = os.path.join(self.ckpt_dir, "best_model.keras")
        self.train_model.save(model_path)

        config = {
            "enc_max_len": self.enc_max_len,
            "dec_max_len": self.dec_max_len,
            "vocab_size": self.vocab_size,
            "emb_dim": self.emb_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
        }
        with open(os.path.join(self.ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, ckpt_dir="weights"):
        """Load a trained model from disk."""
        config_path = os.path.join(ckpt_dir, "config.json")
        model_path = os.path.join(ckpt_dir, "best_model.keras")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run `python main.py train` first."
            )

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        instance = cls(ckpt_dir=ckpt_dir, **config)
        instance.train_model = keras.models.load_model(model_path)
        instance.encoder_model, instance.decoder_model = instance._build_inference_models()
        return instance
