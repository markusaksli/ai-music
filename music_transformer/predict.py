import math

import music21.midi.realtime
import numpy as np
import tensorflow as tf

from music_transformer.convert import midi2idxenc, idxenc2stream
from music_transformer.transformer import MusicGenerator
from music_transformer.vocab import MusicVocab


def load_model(fp):
    return tf.saved_model.load(fp)


def get_song_tokens(vocab):
    return midi2idxenc('../midi_songs/ff4pclov.mid', vocab, add_bos=True, add_eos=False)


def get_middle_c_song(vocab):
    return [vocab.stoi['n100'], vocab.stoi['d10'], vocab.stoi['n109'], vocab.stoi['d10'], vocab.stoi['n112'],
            vocab.stoi['d10']]


if __name__ == '__main__':
    loaded_model = tf.saved_model.load('../trained_models/decoder_only_smaller_1024_mega_ds')
    generator = MusicGenerator(loaded_model)
    created_vocab = MusicVocab.create()

    # tokens_original = midi2idxenc('../midi_songs/costadsol.mid', created_vocab, add_bos=False, add_eos=False)
    # tokens_original = get_middle_c_song(created_vocab)

    gen_len = 1024
    generated = generator.extend_sequence(input_sequence=None, max_generate_len=gen_len)
    print(generated[-(gen_len + 2):])

    generated = idxenc2stream(generated.numpy(), vocab=created_vocab)
    sp = music21.midi.realtime.StreamPlayer(generated)
    sp.play()
