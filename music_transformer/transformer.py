"""
Implementation of a layered Transformer-Decoder model
based on: https://www.tensorflow.org/text/tutorials/transformer
"""

import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out2 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out2, attn_weights_block1


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, att_weight = self.dec_layers[i](x, training, look_ahead_mask)

            attention_weights[f'decoder_layer{i + 1}'] = att_weight

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class TransformerDecoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_target, rate=0.1):
        super().__init__()

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, tar, training):
        # Keras models prefer if you pass all your inputs in the first argument

        look_ahead_mask = self.create_masks(tar)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, training, look_ahead_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def create_masks(self, tar):
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return look_ahead_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


class TransformerMusicGenerator(tf.Module):
    def __init__(self, transformer, max_len):
        self.transformer = transformer
        self.max_len = max_len

    def predict_token(self, sequence):
        input_seq = sequence[-self.max_len:]
        input_seq = tf.expand_dims(input_seq, axis=0)
        predictions, _ = self.transformer(input_seq, training=False)
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

    def beam_search(self, sequence, gen_len, beam_width, creativity):
        assert isinstance(sequence, tf.Tensor)
        if sequence[0] != tf.cast(1, tf.int32):
            sequence = self.add_start_tokens(sequence)

        # First step: generate the beam by predicting the next element top k options
        predictions = self.predict_token(sequence)
        vocab_len = tf.shape(predictions)[0]
        predictions = self.apply_creativity(predictions, creativity)
        logits, indicies = tf.math.top_k(predictions, beam_width, sorted=False)
        softmax = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]

        beams = tf.TensorArray(dtype=tf.int32, size=beam_width, dynamic_size=False)
        beam_probs = tf.TensorArray(dtype=tf.float32, size=beam_width, dynamic_size=False)
        for i in tf.range(beam_width):
            beam_probs = beam_probs.write(i, tf.math.log(softmax[i]))
            beams = beams.write(i, tf.concat([sequence, [indicies[i]]], 0))

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
        for _ in tf.range(gen_len - 1):
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
        _, attention_weights = self.transformer(tf.expand_dims(beams[top_idx][-(gen_len + 1):-1], axis=0), training=False)
        return beams[top_idx], attention_weights

    def iterative_search(self, sequence, gen_len, greedy, top_k_notes, top_k_durations, top_k_offset, creativity):
        assert isinstance(sequence, tf.Tensor)
        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        if sequence[0] != tf.cast(1, tf.int32):
            sequence = self.add_start_tokens(sequence)
        original_len = tf.shape(sequence)[0]
        sequence_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        for i in tf.range(original_len):
            sequence_array = sequence_array.write(i, sequence[i])

        for i in tf.range(gen_len):
            input_seq = sequence_array.stack()
            # tf.print(input_seq)
            predictions = self.predict_token(input_seq)
            # tf.print(predictions)

            most_likely = self.most_likely_prediction(predictions)
            # Creativity bias
            predictions = self.apply_creativity(predictions, creativity)
            # tf.print(predictions)

            if greedy:
                # Resample after creativity bias
                most_likely = self.most_likely_prediction(predictions)
                sequence_array = sequence_array.write(original_len + i, most_likely)

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
                sequence_array = sequence_array.write(original_len + i, predicted_id)
                # tf.print(tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0])
            # tf.print(sequence_array.stack())
        output = sequence_array.stack()

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer(tf.expand_dims(output[-(self.max_len + 1):-1], axis=0), training=False)

        return output, attention_weights


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
            input_sequence: np.array of integers, the token indices for the starting prompt. Set to None to generate without a prompt.
            max_generate_len: Integer, the number of tokens to be generated after prompt.

            search: 'greedy' (default and fastest), 'top_k' (more random and controllable), or 'beam' (most "accurate" but slowest)
            top_k_notes: Number of top possibilities to sample from when choosing a predicted note pitch token.
            top_k_durations: Number of top possibilities to sample from when choosing a duration token.
            top_k_offset: Number of top possibilities to remove from sampling to bias towards unlikely sequences during top_k search.
            beam_width: Number of sequences iterated on during Beam Search. For regular systems it isn't recommended to use anything over 10.
            creativity: Will bias generation towards less separators and more notes by multiplying the separator token log probabilty by 1 - creativity / 1000. This needs to be passed as an integer argument so it needs to be between 0 and 1000 inclusive.
        """
        assert 1000 >= creativity >= 0
        assert top_k_notes - top_k_offset > 1
        assert top_k_durations - top_k_offset > 1
        if input_sequence is None:
            input_sequence = np.array([1, 0])
        input_sequence = np.array(input_sequence).astype('int32')
        if search == 'beam':
            beam_arg_arr_1 = np.array([max_generate_len, beam_width] + [0] * (len(input_sequence) - 2)).astype('int32')
            beam_arg_arr_2 = np.array([creativity, 0] + [0] * (len(input_sequence) - 2)).astype('int32')
            return self.model.beam_search([input_sequence, beam_arg_arr_1, beam_arg_arr_2])[0]
        arg_arr_1 = np.array([max_generate_len, top_k_notes] + [0] * (len(input_sequence) - 2)).astype('int32')
        arg_arr_2 = np.array([top_k_durations, top_k_offset] + [0] * (len(input_sequence) - 2)).astype('int32')
        arg_arr_3 = np.array([creativity, 0] + [0] * (len(input_sequence) - 2)).astype('int32')
        if search == 'greedy':
            return self.model.greedy_search([input_sequence, arg_arr_1, arg_arr_2, arg_arr_3])[0]
        if search == 'top_k':
            return self.model.random_search([input_sequence, arg_arr_1, arg_arr_2, arg_arr_3])[0]

    def save(self, fp):
        tf.saved_model.save(self.model, export_dir=fp)
