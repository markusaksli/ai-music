# Music Generation With a Transformer Decoder Model
This repository contains **TensorFlow** code for training a vanilla layered [Transformer](https://www.tensorflow.org/text/tutorials/transformer) Decoder model using MIDI tokenization code from [musicautobot](https://github.com/bearpelican/musicautobot). There is also a trained model that was trained on a subset of [Final Fantasy OST MIDI files](https://github.com/Skuldur/Classical-Piano-Composer/tree/master/midi_songs).

## Generator GUI

The trained model can be played around with by running [music_generator_gui.py](https://github.com/markusaksli/ai-music/blob/main/music_generator_gui.py).

![Generator](https://github.com/markusaksli/ai-music/blob/main/samples/images/generator.png)

#### Requirements
The requirements to run the music generator or to train a new model are in [requirements.txt](https://github.com/markusaksli/ai-music/blob/main/requirements.txt)

The model was trained on a single GTX 1080 Ti and is able to generate about 16 notes in 1.5s using greedy search on the same hardware. This should mean the model can run on most modern GPUs but CPU performance has not been testsed.

#### Basic Controls
The GUI can be used to set the arguments for the generation algorithm and then a new sequence can be generated for the given input (or empty imput). Once a sequence has been generated (which may freeze the GUI and take a while) it will automatically start playing. It can then be extended further or saved to a file.

## Sequence generation algorithms
The trained models can be used in a [TensorFlow Module](https://github.com/markusaksli/ai-music/blob/785e54fef80696f3fc7c505835f08e620fbd59f7/music_transformer/transformer.py#L274-L433) that has different search algorithms for sequence generation. There are three methods of generating new tokens to extend the given sequence using extend_sequence (chosen by the *search* argument in the GUI):

#### Greedy Search:
This is the simplest and fastest method. It will simply sample the most likely token from the last position in the model output and add it back to the input sequence. It will keep iterating until it reaches the desired length or an EOS token.            

#### Top K Sampling (*top_k*):
This method also simply takes the log probabilities for the last token in the model output sequence but instead of simply choosing the most likely one it will sample from the top k logits (based on their probability distribution). The `top_k_notes` and `top_k_durations` arguments can be used to make the sampling window larger (more randomness) or smaller.

If the output is still highly similar to Greedy Search it's likely that the top k softmax distribution looks something like `[0.999, 0.00001, 0.0000005, ...]`. In this case the `top_k_offset` argument can be used to remove the first n indicies from the distribution, making it far more likely to generate sequences that are completely different compared to greedy search.

#### Beam Search:
This method is similar to greedy search but it will give an approximation for the most likely total generated sequence instead of naively sampling the most likely token at each step. To do this it keeps track of `n=beam_width` sequences at each step and considers the next token possibilities for all of them (evaluating `beam_width * vocab_len` probabilies at each step).

At the end of each step it will reduce all the possible continuations for the current `n=beam_width` sequences to the top `k=beam_width` most likely sequences (summing the log probability, which is the same as maximising the multiplied probabilities but more numerically stable). At the end this will give a far bettera pproximation for the most "accurate" continuation of the input sequence compared to Greedy Search (just a lot slower).

**All of these methods can be biased towards generating less rests (SEP tokens between groups of notes played at the same time) and more notes by using the `creativity` argument.**

# Samples
While attempting to train a model to generate novel sequences of music based off of the input we ran into a lot of overfitting due to the relatively small dataset. Previously we attempted to use a classic Transformer Seq2Seq model with both an encoder and decoder. This led to poor results with either underfitting and generating noisy sequences, or completely overfitting and generating a training sequence one-to-one regardless of the input.

https://user-images.githubusercontent.com/54057327/149260274-8340f525-db13-43d9-b87d-16347a23e9a4.mp4

After removing the encoder from the model, sizing it down, increasing the input sequence length (max positional encoding length) and increasing dropout, the model started to come up with quite novel but still tonally coherent sequences.

https://user-images.githubusercontent.com/54057327/149260267-40eeebae-dfa7-4fbf-9a1b-58362e7a2471.mp4

An interesting side-effect was that as the model started to more confidently predict sequences of notes instead of silence, the validation loss during training grew. We still saw very high training accuracy at the end of training, indicating some degree of overfitting however the predicted output was no longer too similar to the training data.

# Training

### [wandb run](https://wandb.ai/markusaksli/ai-music/runs/1dvw6st2)

![Trained Model](https://github.com/markusaksli/ai-music/blob/main/samples/images/run_summary.png) ![Hyperparameters](https://user-images.githubusercontent.com/54057327/149263773-2087fbdd-2503-4932-b18f-20bb532e0463.png)

## Sources and References
**Tokenizing MIDI files, tips on improving generalization**:
- https://github.com/bearpelican/musicautobot
- https://towardsdatascience.com/practical-tips-for-training-a-music-model-755c62560ec2

**TensorFlow 2 Transformer model**
- https://www.tensorflow.org/text/tutorials/transformer

**Transformer XL (suggested in musicautobot but could not sucessfully implement in TensorFlow 2)**
- [paper](https://arxiv.org/abs/1901.02860)
- [author implementation in TF 1.12.0](https://github.com/kimiyoung/transformer-xl/tree/master/tf)
- [used implementation in TF 2](https://github.com/dwdb/transformer-xl)

**Beam Search**
- https://www.youtube.com/watch?v=RLWuzLLSIgw
- https://www.youtube.com/watch?v=gb__z7LlN_4
- https://www.youtube.com/watch?v=ZGUZwk7xIwk

**Music Transformer (did not get to trying to implement their method of relative positional embedding)**
- https://magenta.tensorflow.org/music-transformer
- https://arxiv.org/abs/1809.04281
- [TensorFlow 2 implementation](https://github.com/jason9693/musictransformer-tensorflow2.0)

**Article and dataset that served as the inital inspiration**
- https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
- https://github.com/Skuldur/Classical-Piano-Composer/tree/master/midi_songs
