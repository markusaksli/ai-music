"""
Custom training loop based on: https://www.tensorflow.org/text/tutorials/transformer
"""
import os

import music21.midi.realtime

os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
import math
import time

from music_transformer.convert import create_dataset, idxenc2stream, midi2idxenc
from music_transformer.vocab import MusicVocab
from transformer import *


max_len = 1024
transpose = 6
batch_size = 12

num_layers = 6
d_model = 256
dff = 512
num_heads = 8
dropout_rate = 0.4

EPOCHS = 1000

if __name__ == '__main__':
    created_vocab = MusicVocab.create()
    print(f'vocabulary size: {len(created_vocab)}')
    transformer = TransformerDecoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        vocab_size=len(created_vocab),
        pe_target=max_len,
        rate=dropout_rate)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='validation_loss')
    val_accuracy = tf.keras.metrics.Mean(name='validation_accuracy')

    checkpoint_path = "../model_checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
        tf_gen = TransformerMusicGenerator(transformer, max_len)
        exporter = ExportMusicGenerator(tf_gen)
        # tokens_original = midi2idxenc('../midi_songs_val/balamb.mid', created_vocab, add_bos=False, add_eos=False)
        generator = MusicGenerator(exporter)
        gen_len = 32
        generated = generator.extend_sequence(input_sequence=None, max_generate_len=gen_len)
        print(generated[-(gen_len + 2):])
        generated = idxenc2stream(generated.numpy(), vocab=created_vocab)
        generated.show('midi')
        generated.write('midi', fp='../generated/generated_greed.mid')
        generator.save('../trained_models/decoder_only_smaller_1024_mega_ds')
        quit()

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]


    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, training=True)
            loss = loss_function(tar, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar, predictions))


    train_ds = create_dataset('../midi_songs/', created_vocab, max_len, batch_size, transpose)
    val_ds = create_dataset('../midi_songs_val', created_vocab, max_len, batch_size, transpose, shuffle=False)

    import wandb

    wandb.init(project="ai-music", entity="markusaksli", dir="..\\", config={
        "sequence_length": max_len,
        "batch_size": batch_size,
        "transposition_range": transpose,
        "num_layers": num_layers,
        "d_model": d_model,
        'dff': dff,
        'num_heads': num_heads,
        'dropout_rate': dropout_rate,
        'epochs': EPOCHS,
    })

    gen_tokens = midi2idxenc('../midi_songs_val/balamb.mid', created_vocab, add_bos=True, add_eos=True)[:128]
    prev_train_loss = math.inf
    prev_val_loss = math.inf
    tf_generator = TransformerMusicGenerator(transformer, max_len)
    exporter = ExportMusicGenerator(tf_generator)
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> seq, tar -> seq(shifted +1)
        for (batch, (train_inp, train_tar)) in enumerate(train_ds):
            train_start = time.time()
            train_step(train_inp, train_tar)
            train_time = time.time() - train_start
            # if batch % 5 == 0:
            train_loss_batch = train_loss.result()
            train_accuracy_batch = train_accuracy.result()
            # print(
            #     f'Epoch {epoch + 1} Batch {batch} Loss {train_loss_batch:.4f} Accuracy {train_accuracy_batch:.4f}')
            if not (epoch == 0 and batch == 0):
                wandb.log({"seq_train_time": train_time / batch_size,
                           "batch_loss": train_loss_batch,
                           "batch_accuracy": train_accuracy_batch})

        # Run a validation loop at the end of each epoch.
        for (batch, (val_inp, val_tar)) in enumerate(val_ds):
            val_logits, _ = transformer(val_inp, training=False)
            # Update val metrics
            val_loss_calc = loss_function(val_tar, val_logits)
            val_loss(val_loss_calc)
            val_accuracy(accuracy_function(val_tar, val_logits))
        val_acc_res = val_accuracy.result()
        val_loss_res = val_loss.result()

        train_loss_epoch = train_loss.result()
        train_accuracy_epoch = train_accuracy.result()
        print(
            f'Epoch {epoch + 1}: Loss {train_loss_epoch:.4f}, Valiation loss {val_loss_res:.4f}, Accuracy {train_accuracy_epoch:.4f}, Valiation accuracy {val_acc_res:.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs')
        wandb.log({"loss": train_loss_epoch,
                   "val_loss": val_loss_res,
                   "accuracy": train_accuracy_epoch,
                   "val_accuracy": val_acc_res})

        if (epoch + 1) % 5 == 0:
            if train_loss_epoch < prev_train_loss or val_loss_res < prev_val_loss:
                print(f'#####################{prev_train_loss:.4f} -> {train_loss_epoch:.4f}')
                prev_train_loss = train_loss_epoch
                prev_val_loss = val_loss_res
                gen_time_start = time.time()
                gen = tf_generator.iterative_search(tf.convert_to_tensor(gen_tokens), 32, True, 0, 0, 0, 1.0)[0]
                gen_time = time.time() - gen_time_start
                # Print the 4 last tokens in the generation input as well to check if the output makes sense
                print(f'generated({gen_time:.4f}):\n{gen[-36:]}\n')
                sp = music21.midi.realtime.StreamPlayer(idxenc2stream(gen.numpy(), created_vocab))
                sp.play(blocked=False)
                wandb.log({'gen_time': gen_time})
                if train_accuracy_epoch > 0.6:
                    ckpt_save_path = ckpt_manager.save()
                    print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
