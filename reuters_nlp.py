"""
Adapted from: https://tensorflow.github.io/tensor2tensor/new_problem.html
Mike Holcomb
Reuters 5-gram example for CS6320
"""
import os
import re

# from gutenberg import acquire
# from gutenberg import cleanup

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensorflow.keras.datasets import reuters

vocab_size=30000
N = 5 # Prefix length

@registry.register_problem
class ReutersNlp(text_problems.Text2TextProblem):
    """Predict next word of 6 gram from prefix"""

    @property
    def approx_vocab_size(self):
        return vocab_size  # ~8k

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def vocab_type(self):
      return text_problems.VocabType.TOKEN

    @property
    def vocab_filename(self):
        return "vocab.10k.txt"

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        vocab_filename = os.path.join(data_dir, self.vocab_filename)
        word_index = self._get_word_index()

        idx = 0
        with open(vocab_filename,"w") as vocab:
            # for k, v in word_index.items():
            for k, v in sorted(word_index.items(), key=lambda x: x[1]):
                while idx < v:
                  vocab.write("unusued{}\n".format(idx))
                  idx += 1

                vocab.write("{}\n".format(k))
                idx+=1

                # vocab.write("{}, {}\n".format(k, v))

        encoder = text_encoder.TokenTextEncoder(vocab_filename,
                                                replace_oov=self.oov_token)

        return encoder

    def _get_word_index(self):
        word_index = reuters.get_word_index()
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3

        return word_index

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        # del data_dir
        del tmp_dir
        del dataset_split

        (train_data, _), (test_data, _) = reuters.load_data(num_words=vocab_size, seed=1337, test_split=0.2)

        word_index = self._get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

        # for seq in test_data:
        for seq in train_data:
            idx = 0
            while idx + N < len(seq) - 1:
                yield {
                  "inputs" : " ".join([reverse_word_index.get(i, '?') for i in seq[idx:idx+N]]),
                  "targets" : str(reverse_word_index.get(seq[idx + N],'?')),
                }
                idx += 1

        # books = [
        #     # bookid, skip N lines
        #     (19221, 223),
        #     (15553, 522),
        # ]
        #
        # for (book_id, toskip) in books:
        #     text = cleanup.strip_headers(acquire.load_etext(book_id)).strip()
        #     lines = text.split("\n")[toskip:]
        #     prev_line = None
        #     ex_count = 0
        #     for line in lines:
        #         # Any line that is all upper case is a title or author name
        #         if not line or line.upper() == line:
        #             prev_line = None
        #             continue
        #
        #         line = re.sub("[^a-z]+", " ", line.strip().lower())
        #         if prev_line and line:
        #             yield {
        #                 "inputs": prev_line,
        #                 "targets": line,
        #             }
        #             ex_count += 1
        #         prev_line = line

@registry.register_hparams
def transformer_reuters():
  hparams = transformer.transformer_base()
  hparams.num_hidden_layers = 4 # 3
  hparams.hidden_size = 128 # 64
  hparams.filter_size = 256 # 512
  hparams.num_heads = 4 # 8
  hparams.attention_dropout = 0.6
  hparams.layer_prepostprocess_dropout = 0.6
  hparams.learning_rate = 0.05
  return hparams
