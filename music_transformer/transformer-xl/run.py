import os

os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
import numpy as np
import tensorflow as tf
import time
from model import TransformerXL, get_generator
import sys

# 'Description: Transformer-XL Simplified Version.'
# vocabulary file
from music_transformer.convert import create_dataset, midi2idxenc, idxenc2stream
from music_transformer.vocab import MusicVocab

VOCAB_FILE = 'data/poetry/vocab.pkl'
# dataset paths
TRAIN_DATA_PATH = '../../midi_songs'
VAL_DATA_PATH = '../../midi_songs_val'
# path for training and valid output, such as save model
OUTPUT_PATH = 'output/'
# song transpose range
TRANSPOSE = 4
BATCH_SIZE = 32
# target length, or sequence length
SEQ_LEN = 256
# memory length
MEM_LEN = 256
# word embeeding size
EMBEDDING_SIZE = 128
# multihead attetion hidden size
HIDDEN_SIZE = 128
# feed forward network hidden size
FFN_SIZE = 1024
# number of heads of multiheads
NUM_HEADS = 8
# number of layers of multihead attention
N_LAYER = 8
DROPOUT_RATE = 0.1
# wheather the bias of each layer of relative multihead attention is different or not
UNTIE_REL_BIAS = True
# inital learning rate
LEARNING_RATE = 0.001
# minimal learning rate
MIN_LEARNING_RATE = 0.004
# clips values of multiple tensors by the ratio of the sum of their norms
CLIP_NORM = 0.25

EPOCHS = 1000
# number of steps between save model
SAVE_EPOCH = 10
# number of steps between verify model
VALID_EPOCH = 5
WARMUP_STEPS = 0
LR_STEPS = 200000


class CosineDecayWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, init_lr, steps, warmup_steps, min_lr):
        super(CosineDecayWarmup, self).__init__()

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.cosine_decay = tf.keras.experimental.CosineDecay(
            init_lr, steps - warmup_steps, min_lr)

    def __call__(self, step):
        linear_increase = self.init_lr * tf.cast(step, tf.float32) / (
                tf.cast(self.warmup_steps, tf.float32) + 1e-5)
        cosine_decay = self.cosine_decay(step)
        return tf.cond(pred=step <= self.warmup_steps,
                       true_fn=lambda: linear_increase,
                       false_fn=lambda: cosine_decay)

    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'init_lr': self.init_lr
        }


def model_fn():
    model = TransformerXL(n_vocab=len(vocab), d_embed=EMBEDDING_SIZE,
                          d_model=HIDDEN_SIZE, d_ff=FFN_SIZE, q_len=SEQ_LEN,
                          m_len=MEM_LEN,
                          num_heads=NUM_HEADS, n_layer=N_LAYER, dropout_rate=DROPOUT_RATE,
                          untie_rel_bias=UNTIE_REL_BIAS)

    return model


vocab = MusicVocab.create()
model = model_fn()


def loss_function(labels, logits):
    """损失函数"""
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def train_step(inputs, labels, optimizer, mems):
    """训练一个batch"""
    with tf.GradientTape() as tape:
        logits, new_mems = model(inputs, mems=mems, training=True)
        loss = loss_function(labels, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    clipped, gnorm = tf.clip_by_global_norm(gradients, CLIP_NORM)
    optimizer.apply_gradients(zip(clipped, model.trainable_variables))

    return loss, new_mems


def train():
    """模型训练"""
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, OUTPUT_PATH, max_to_keep=3, checkpoint_name='xl-ckpt')

    learning_rate = CosineDecayWarmup(LEARNING_RATE, LR_STEPS, WARMUP_STEPS,
                                      MIN_LEARNING_RATE)
    # optimizer = tf.keras.optimizers.Adamax()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_ds = create_dataset('../midi_songs/', vocab, SEQ_LEN, BATCH_SIZE, TRANSPOSE, shuffle=False)
    val_ds = create_dataset('../midi_songs_val', vocab, SEQ_LEN, BATCH_SIZE, TRANSPOSE, shuffle=False)
    mems = None

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')

    import wandb

    wandb.init(project="ai-music", entity="markusaksli", dir="../..\\", config={
        "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "transposition_range": TRANSPOSE,
        "num_layers": N_LAYER,
        "d_model": HIDDEN_SIZE,
        'dff': FFN_SIZE,
        'd_embed': EMBEDDING_SIZE,
        'mem_len': MEM_LEN,
        'num_heads': NUM_HEADS,
        'dropout_rate': DROPOUT_RATE,
        'clip_norm': CLIP_NORM,
        'lr_steps': LR_STEPS,
        'warmup_steps': WARMUP_STEPS,
        'learning_rate': LEARNING_RATE,
        'min_learning_rate': MIN_LEARNING_RATE,
    })

    steps = 0
    gen_tokens = midi2idxenc('../../midi_songs_val/balamb.mid', vocab=vocab, add_eos=False, add_bos=True)[:128]
    for epoch in range(EPOCHS):
        old_time = time.time()
        for (batch, (inputs, labels)) in enumerate(train_ds):
            train_time_start = time.time()
            loss, mems = train_step(inputs, labels, optimizer=optimizer, mems=mems)
            train_time = time.time() - train_time_start
            train_loss(loss)

            train_loss_batch = train_loss.result()
            if steps > 1:
                wandb.log({"seq_train_time": train_time / BATCH_SIZE, "batch_loss": train_loss_batch,
                           "learning_rate": learning_rate(steps)})
            steps += 1

        train_loss_res = train_loss.result()
        train_loss.reset_states()
        print('{} epoch: {} | loss: {:.4f} | lr: {} | {:.2f}s'.format(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            epoch,
            train_loss_res,
            learning_rate(steps),
            (time.time() - old_time)))

        wandb.log({"loss": train_loss_res, "learning_rate": learning_rate(steps)})

        if epoch % VALID_EPOCH == 0:
            evaluate(val_ds, val_loss)
            val_loss_res = val_loss.result()
            val_loss.reset_states()
            print(f'====\nvalidation loss: {val_loss_res:.3f}\n====')
            print('generated:')
            gen_time_start = time.time()
            print(inference(gen_tokens, max_len=32, restore=False))
            gen_time = time.time() - gen_time_start
            wandb.log({"val_loss": val_loss_res, 'gen_time': gen_time})

        if epoch % SAVE_EPOCH == 0:
            print('saving checkpoint for epoch {} at {}'.format(
                steps, ckpt_manager.save()))


def evaluate(valid_dataset, loss_obj):
    """模型验证"""
    mems = [None] * model.n_layer
    for (step, (inputs, labels)) in enumerate(valid_dataset):
        logits, mems = model(inputs, mems=mems, training=False)
        loss = loss_function(labels, logits)
        loss_obj(loss)


def inference(inp_seq=None, tgt_len=SEQ_LEN, mem_len=MEM_LEN, max_len=128, restore=True):
    def fn(seq):
        mems = [None] * model.n_layer
        x = seq[-rel_len:]
        generated_tokens = []
        for i in range(max_len):
            x = tf.constant([x], dtype=tf.int32)
            output, mems = model(x, mems=mems, training=False)
            x = tf.argmax(output[:, -1], axis=-1).numpy()
            # early stop when the eos symbol has generated
            if x[0] == 2:
                break
            generated_tokens.append(x[0])
        return generated_tokens

    # 相对位置编码的长度
    rel_len = model.q_len + model.m_len
    # memory length of inference
    model.m_len = mem_len
    model.q_len = tgt_len

    if restore:
        checkpoint_path = tf.train.latest_checkpoint(OUTPUT_PATH)
        print('restoring model from {}'.format(checkpoint_path))
        tf.train.Checkpoint(model=model).restore(checkpoint_path)

    if inp_seq is None:
        inp_seq = np.array([1, 0])
        print('primed with empty sequence')
    return fn(inp_seq)


if __name__ == '__main__':
    train()
    # vocab = MusicVocab.create()
    # idxenc = midi2idxenc('../midi_songs/ff4pclov.mid', vocab=vocab, add_eos=False, add_bos=True)[:128]
    # generated = np.append(idxenc, inference(idxenc, mem_len=128, max_len=512))
    # print(generated)
    # mid = idxenc2stream(generated, vocab)
    # mid.show('midi')
