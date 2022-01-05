"""
Attempt at using a TensorFlow 2 compatible TransformerXL model (did not work well so remains unused currently)
source: https://github.com/dwdb/transformer-xl
"""

import numpy as np
import tensorflow as tf

INITIALIZER = tf.keras.initializers.RandomNormal(stddev=0.01)

def relative_mask(q_len, m_len):
    """相对位置掩码，当前位置左侧为1、右侧为0"""
    mask = tf.sequence_mask(tf.range(1, q_len + 1), q_len, dtype=tf.float32)
    mask = tf.pad(mask, [[0, 0], [m_len, 0]], constant_values=1)
    return mask


def positional_embedding(k_len, d_model):
    """绝对位置编码"""
    inv_freq = 1. / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    pos_seq = tf.range(k_len - 1, -1, -1.0)
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    return pos_emb[None, :, :]


def point_wise_feed_forward_network(d_model, d_ff):
    """前馈网络"""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu',
                              kernel_initializer=INITIALIZER, name='ffn1'),
        tf.keras.layers.Dense(d_model, kernel_initializer=INITIALIZER, name='ffn2')
    ])


class RelMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dropout_rate):
        super(RelMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_depth = self.d_model // self.num_heads

        self.w_head = tf.keras.layers.Dense(
            3 * d_model, use_bias=False, kernel_initializer=INITIALIZER)
        self.r_head = tf.keras.layers.Dense(
            d_model, use_bias=False, kernel_initializer=INITIALIZER)

        self.dense = tf.keras.layers.Dense(
            d_model, use_bias=False, kernel_initializer=INITIALIZER)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    @staticmethod
    def relative_shift(x):
        """行元素左移，结合mask实现相对位置编码。移动步数为n-i（n为行总数，i为行号，首行为1）。
        """
        x_size = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, (x_size[0], x_size[1], x_size[3] + 1, x_size[2]))
        x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)
        return x

    def call(self, inputs, pos_emb, r_w_bias, r_r_bias, mems, training, **kwargs):
        """
        inputs: shape=(batch_size, q_len, d_model)
        pos_emb: shape=(1, k_len, d_model)
        u: shape=(num_heads, d_depth)
        v: shape=(num_heads, d_depth)
        mems: shape=(batch_size, m_len, d_model)
        attn_mask: shape=(m_len + q_len, q_len)
        """
        batch_size = tf.shape(inputs)[0]
        q_len = tf.shape(inputs)[1]
        # 拼接缓存
        if mems is None:
            cat = inputs
        else:
            cat = tf.concat((mems, inputs), axis=1)
        cat = self.dropout1(cat, training=training)
        # 拼接后的上下文长度，k_len = m_len + q_len
        k_len = tf.shape(cat)[1]
        m_len = k_len - q_len
        # shape=(1, k_len, d_model)
        pos_emb = pos_emb[:, -k_len:]
        pos_emb = self.dropout2(pos_emb, training=training)

        w_heads = tf.reshape(self.w_head(cat), (
            batch_size, k_len, 3 * self.num_heads, self.d_depth))
        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, axis=2)
        # shape=(batch_size, q_len, num_heads, d_depth)
        w_head_q = w_head_q[:, -q_len:]

        # shape=(batch_size, num_heads, q_len, k_len)
        ac = tf.einsum('bqnd,bknd->bnqk', w_head_q + r_w_bias, w_head_k)
        r_head_k = tf.reshape(self.r_head(pos_emb), (k_len, self.num_heads, self.d_depth))
        bd = tf.einsum('bqnd,knd->bnqk', w_head_q + r_r_bias, r_head_k)
        bd = self.relative_shift(bd)

        attn_mask = relative_mask(q_len, m_len)
        # shape=(batch_size, num_heads, q_len, k_len)
        attn_score = (ac + bd) / (self.d_depth ** 0.5)
        attn_score = attn_score * attn_mask - 1e30 * (1. - attn_mask)
        attn_score = tf.nn.softmax(attn_score, axis=-1)

        attn_vec = tf.einsum('bnqk,bknd->bqnd', attn_score, w_head_v)
        attn_vec = tf.reshape(attn_vec, (batch_size, q_len, self.d_model))

        attn_out = self.dense(attn_vec)
        return attn_out


class TransformerLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, d_ff, num_heads, dropout_rate):
        super(TransformerLayer, self).__init__()

        self.rel_multihead_attn = RelMultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate)
        # feed forward network
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        # layer normalization
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, pos_emb, r_w_bias, r_r_bias, mems, training, **kwargs):
        attn_out = self.rel_multihead_attn(inputs=inputs, pos_emb=pos_emb,
                                           r_w_bias=r_w_bias, r_r_bias=r_r_bias,
                                           mems=mems, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layer_norm1(inputs + attn_out)

        ffn_out = self.ffn(out1, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layer_norm2(out1 + ffn_out)
        return out2


class TransformerXL(tf.keras.Model):

    def __init__(self, n_vocab, d_embed, d_model, d_ff, q_len, m_len, num_heads,
                 n_layer, dropout_rate, untie_rel_bias):
        super(TransformerXL, self).__init__()
        self.d_embed = d_embed
        self.d_model = d_model

        self.q_len = q_len
        self.m_len = m_len
        self.n_layer = n_layer
        self.untie_rel_bias = untie_rel_bias

        # word embedding
        self.embedding = tf.Variable(INITIALIZER((n_vocab, d_embed)), name='embedding')
        # word embedding size to model size
        self.projection = tf.Variable(INITIALIZER((d_embed, d_model)), name='projection')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.pos_emb = positional_embedding(q_len + m_len, d_model)

        shape = (2, n_layer if untie_rel_bias else 1, num_heads, d_model // num_heads)
        self.rw_bias = tf.Variable(INITIALIZER(shape), name='rw_bias')
        self.logit_bias = tf.Variable(tf.zeros((n_vocab,)), name='logit_bias')

        self.multihead_layers = []
        for i in range(n_layer):
            layer = TransformerLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                                     dropout_rate=dropout_rate)
            self.multihead_layers.append(layer)

    def cache_mems(self, cur_out, pre_mem):
        if self.m_len is None or self.m_len <= 0:
            return None
        if pre_mem is None:
            new_mem = cur_out
        else:
            new_mem = tf.concat((pre_mem, cur_out), axis=1)
        return tf.stop_gradient(new_mem[:, -self.m_len:])

    def call(self, inputs, mems=None, training=False, **kwargs):
        new_mems = []
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        x = tf.matmul(x, self.projection)

        if mems is None:
            mems = [None] * self.n_layer

        for i in range(self.n_layer):
            new_mems.append(self.cache_mems(x, mems[i]))
            j = i if self.untie_rel_bias else 0
            x = self.multihead_layers[i](inputs=x,
                                         pos_emb=self.pos_emb,
                                         r_w_bias=self.rw_bias[0][j],
                                         r_r_bias=self.rw_bias[1][j],
                                         mems=mems[i],
                                         training=training)

        x = self.dropout1(x, training=training)
        # share embedding parameters with inputs
        # shape=(batch_size, seq_len, d_embed)
        # tf.einsum('bik,jk->bij', x, self.projection)
        x = tf.matmul(x, self.projection, transpose_b=True)
        # shape=(batch_size, seq_len, n_vocab)
        x = tf.matmul(x, self.embedding, transpose_b=True) + self.logit_bias

        return x, new_mems


class TransformerMusicGenerator(tf.Module):
    def __init__(self, transformer):
        self.transformer = transformer
        self.mems = None

    def predict_token(self, sequence, mems, update_mems=False):
        # tf.print(input_seq)
        # tf.print(target_seq)

        input_seq = tf.expand_dims(sequence, axis=0)

        predictions, new_mems = self.transformer(input_seq, mems=mems, training=False)
        if update_mems:
            self.mems = new_mems
        return predictions[0][-1]

    def most_likely_prediction(self, predictions):
        return tf.argmax(predictions, axis=-1, output_type=tf.int32)

    def add_start_tokens(self, sequence):
        return tf.concat([[1, 0], sequence], 0)

    def apply_creativity(self, predictions, creativity):
        # Epsilon compariston to avoid remapping if it isn't likely to even do anything
        if creativity > 0.99:
            return predictions
        len_pred = tf.shape(predictions)[0]

        remapped = tf.TensorArray(dtype=tf.float32, size=len_pred, dynamic_size=False)
        for i in tf.range(len_pred):
            probability = predictions[i]
            if i == 3:
                # tf.print(probability)
                probability = probability * creativity
                # tf.print(probability)
            remapped = remapped.write(i, probability)

        return remapped.stack()

    def beam_search(self, sequence, max_len, beam_width, creativity):
        assert isinstance(sequence, tf.Tensor)
        input_seq = self.add_start_tokens(sequence)

        # First step: generate the beam by predicting the next element top k options
        predictions = self.predict_token(input_seq)
        vocab_len = tf.shape(predictions)[0]
        predictions = self.apply_creativity(predictions, creativity)
        logits, indicies = tf.math.top_k(predictions, beam_width, sorted=False)
        softmax = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]

        beams = tf.TensorArray(dtype=tf.int32, size=beam_width, dynamic_size=False)
        beam_probs = tf.TensorArray(dtype=tf.float32, size=beam_width, dynamic_size=False)
        for i in tf.range(beam_width):
            beam_probs = beam_probs.write(i, tf.math.log(softmax[i]))
            beams = beams.write(i, tf.concat([input_seq, [indicies[i]]], 0))

        beams = beams.stack()
        beam_probs = tf.transpose(beam_probs.stack())

        # Iteration step:
        # 1. predict the top k options for the next token for each of our beam sequences
        # 2. calculate the probability of the beam sequence with the new token
        #       p = p_old + log(p_new)
        # (this is the same as maximising p = p_old * p_new but more numerically stable)
        # 3. add all of the possible new sequences and probabilities to a buffer (k squared options)
        # 3. choose the top k possible extended sequences as our new beam sequences
        # 4. keep track of the new sequence probabilities to calculate them in the next step
        for _ in tf.range(max_len - 1):
            step_seqs = tf.TensorArray(dtype=tf.int32, size=beam_width * vocab_len, dynamic_size=False)
            step_probs = tf.TensorArray(dtype=tf.float32, size=beam_width * vocab_len, dynamic_size=False)

            for i in tf.range(beam_width):
                beam_step_seq = beams[i]

                # We only add the top k options to the buffer because even if we choose the next
                # top k beam sequences only from the ones that extend this beam sequence we need
                # at max to keep track of the top k new possible sequences, not all of them (vocab size).
                predictions = self.predict_token(beam_step_seq)
                predictions = self.apply_creativity(predictions, creativity)
                softmax = tf.keras.activations.softmax(tf.expand_dims(predictions, 0))[0]
                for j in tf.range(vocab_len):
                    step_seqs = step_seqs.write(i * vocab_len + j, tf.concat([beam_step_seq, [j]], 0))
                    step_probs = step_probs.write(i * vocab_len + j, beam_probs[i] + tf.math.log(softmax[j]))
                # tf.print(step_seqs.stack())
                # tf.print(tf.transpose(step_probs.stack()))

            # Top k prediction indicies
            next_probs, next_seqs_indicies = tf.math.top_k(tf.transpose(step_probs.stack()), beam_width)
            beam_probs = next_probs
            beams = step_seqs.gather(next_seqs_indicies)
            # tf.print(beams)
            # tf.print(beam_probs)

        # Final step: return the sequence with the largest probability after iterating
        top_idx = tf.argmax(beam_probs)
        return beams[top_idx]

    def iterative_search(self, sequence, max_len, greedy, top_k_notes, top_k_durations, top_k_offset, creativity):
        assert isinstance(sequence, tf.Tensor)
        self.mems = None
        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        original_len = tf.shape(sequence)[0]
        sequence_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        for i in tf.range(original_len):
            sequence_array = sequence_array.write(i + 2, sequence[i])

        for i in tf.range(max_len):
            input_seq = sequence_array.stack()
            # tf.print(input_seq)
            predictions = self.predict_token(input_seq, update_mems=True)
            # tf.print(predictions)

            most_likely = self.most_likely_prediction(predictions)
            # Creativity bias
            predictions = self.apply_creativity(predictions, creativity)
            # tf.print(predictions)

            if greedy:
                # Resample after creativity bias
                most_likely = self.most_likely_prediction(predictions)
                sequence_array = sequence_array.write(2 + original_len + i, most_likely)

                # tf.print(most_likely)
                # logits, indicies = tf.math.top_k(predictions, 3, sorted=True)
                # tf.print(indicies)
                # tf.print(tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0])
            else:
                is_note = most_likely < 132
                if is_note:
                    k = top_k_notes
                else:
                    k = top_k_durations
                logits, indicies = tf.math.top_k(predictions, k, sorted=True)

                # Remove the most likely entries for more unlikely predictions
                logits = logits[top_k_offset:]
                indicies = indicies[top_k_offset:]
                # tf.print(indicies)
                # tf.print(tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0])
                predicted_index = tf.random.categorical([logits], 1, dtype=tf.int32)[0][0]
                predicted_id = indicies[predicted_index]

                # We want randomness but also want to avoid generating an incoherent sequence
                if (is_note and predicted_id >= 132) or (not is_note and predicted_id < 132):
                    predicted_id = most_likely
                # tf.print(predicted_id)

                # We are looking for interesting sequences so we should ignore special tokens
                if predicted_id < tf.cast(3, tf.int32):
                    predicted_id = tf.cast(3, tf.int32)
                sequence_array = sequence_array.write(2 + original_len + i, predicted_id)
                # tf.print(tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0])
            # tf.print(sequence_array.stack())
        return sequence_array.stack()


class ExportMusicGenerator(tf.Module):
    def __init__(self, generator):
        self.generator = generator

    @tf.function(input_signature=[tf.TensorSpec(shape=[4, None], dtype=tf.int32)])
    def greedy_search(self, compound):
        creativity = self.calc_creativity(compound[3][0])
        return self.generator.iterative_search(compound[0], compound[1][0], True, 0, 0, 0, creativity)

    @tf.function(input_signature=[tf.TensorSpec(shape=[4, None], dtype=tf.int32)])
    def random_search(self, compound):
        creativity = self.calc_creativity(compound[3][0])
        return self.generator.iterative_search(compound[0], compound[1][0], False, compound[1][1], compound[2][0],
                                               compound[2][1], creativity)

    @tf.function(input_signature=[tf.TensorSpec(shape=[3, None], dtype=tf.int32)])
    def beam_search(self, compound):
        creativity = self.calc_creativity(compound[2][0])
        return self.generator.beam_search(compound[0], compound[1][0], compound[1][1], creativity)

    def calc_creativity(self, creativity):
        return 1.0 - tf.cast(creativity, tf.float32) / 1000.0


class MusicGenerator:
    """A helper class to generate notes for a given input sequence using a trained model.

    There are three methods of generating new tokens to extend the given sequence using extend_sequence (chosen by the
    *search* argument):

        **Greedy Search** ('greedy'):
            This is the simplest and fastest method. It will simply sample the most likely token from the last position
            in the model output and add it back to the input sequence. It will keep iterating until it reaches the
            desired length or an EOS token.
        **Top K Sampling** ('top_k'):
            This method also simply takes the log probabilities for the last token in the model output sequence but
            instead of simply choosing the most likely one it will sample from the top k logits (based on their
            probability distribution). The *top_k_notes* and *top_k_durations* arguments can be used to make the
            sampling window larger (more randomness) or smaller.

            If the output is still highly similar to Greedy Search it's likely that the top k softmax distribution looks
            something like [0.999, 0.00001, 0.0000005, ...]. In this case the *top_k_offset* argument can be used to
            remove the first n indicies from the distribution, making it far mor likely to generate unlikely and more
            unpredictable sequences.
        **Beam Search** ('beam'):
            This method is similar to greedy search but it will give an approximation for the most likely total
            generated sequence instead of naively sampling the most likely token at each step. To do this it keeps track
            of n=*beam_width* sequences at each step and considers the next token possibilities for all of them
            (evaluating *beam_width* * *vocab_len* probabilies at each step).

            At the end of each step it will reduce all the possible continuations for the current n=*beam_width*
            sequences to the top k=*beam_width* most likely sequences (summing the log probability, which is the same as
            maximising the multiplied probabilities but more numerically stable). At the end this will give a far better
            approximation for the most "accurate" continuation of the input sequence compared to Greedy Search (just a
            lot slower).

        All of these methods can be biased towards generating less rests (SEP tokens between groups of notes played at
        the same time) and more notes by using the *creativity* argument.

    Arguments:
        model: Trained model used for prediction.
    """

    def __init__(self, model):
        self.model = model

    def extend_sequence(self, input_sequence, max_generate_len, search='greedy', top_k_notes=3, top_k_durations=3,
                        top_k_offset=0, beam_width=3, creativity=0):
        """
        Extend the given input sequence by *max_generate_len* tokens.

        Arguments:
            input_sequence: np.array of integers, the token indices for the starting prompt.
            max_generate_len: Integer, the number of tokens to be generated after prompt.

            search: 'greedy' (default and fastest), 'top_k' (more random and controllable), or 'beam' (most "accurate" but slowest)
            top_k_notes: Number of top possibilities to sample from when choosing a predicted note pitch token.
            top_k_durations: Number of top possibilities to sample from when choosing a duration token.
            top_k_offset: Number of top possibilities to remove from sampling to bias towards unlikely sequences during top_k search.
            beam_width: Number of sequences iterated on during Beam Search. For regular systems it isn't recommended to use anything over 10.
            creativity: Will bias generation towards less separators and more notes by multiplying the separator token log probabilty by 1 - creativity / 1000. This needs to be passed as an integer argument so it needs to be between 0 and 1000 inclusive.
        """
        assert 1000 >= creativity >= 0
        input_sequence = np.array(input_sequence).astype('int32')
        if search == 'beam':
            beam_arg_arr_1 = np.array([max_generate_len, beam_width] + [0] * (len(input_sequence) - 2)).astype('int32')
            beam_arg_arr_2 = np.array([creativity, 0] + [0] * (len(input_sequence) - 2)).astype('int32')
            return self.model.beam_search([input_sequence, beam_arg_arr_1, beam_arg_arr_2])
        arg_arr_1 = np.array([max_generate_len, top_k_notes] + [0] * (len(input_sequence) - 2)).astype('int32')
        arg_arr_2 = np.array([top_k_durations, top_k_offset] + [0] * (len(input_sequence) - 2)).astype('int32')
        arg_arr_3 = np.array([creativity, 0] + [0] * (len(input_sequence) - 2)).astype('int32')
        if search == 'greedy':
            return self.model.greedy_search([input_sequence, arg_arr_1, arg_arr_2, arg_arr_3])[0]
        if search == 'top_k':
            return self.model.random_search([input_sequence, arg_arr_1, arg_arr_2, arg_arr_3])[0]

    def save(self, fp):
        tf.saved_model.save(self.model, export_dir=fp)


def get_generator(model):
    return MusicGenerator(ExportMusicGenerator(TransformerMusicGenerator(model)))


if __name__ == '__main__':
    n_vocab = 1000
    d_embed = 128
    d_model = 128
    d_ff = 512
    q_len = 16
    m_len = 32
    num_heads = 8
    n_layer = 6
    dropout_rate = 0.1
    batch_size = 8
    mem_transformer = TransformerXL(n_vocab=n_vocab,
                                    d_embed=d_embed,
                                    d_model=d_model,
                                    d_ff=d_ff,
                                    q_len=q_len,
                                    m_len=m_len,
                                    num_heads=num_heads,
                                    n_layer=n_layer,
                                    dropout_rate=dropout_rate,
                                    untie_rel_bias=True)
    inputs = tf.reshape(tf.range(batch_size * q_len), shape=(batch_size, q_len))
    output1, mems1 = mem_transformer(inputs, training=False)
    mem_transformer.mems = mems1
    output2, mems2 = mem_transformer(inputs, training=False)
    print(output1[0][0])
    print(output2[0][0])
