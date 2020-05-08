"""
https://github.com/salesforce/awd-lstm-lm

. ./get_lm_data.sh  # download datasets

python train-language-modeling.py \
    --model LSTM --nlayers 3 --emsize 200 --nhid 1000 --bptt 150 \
    --data data/pennchar --train_bs 128 \
    --optimizer adam --lr 2e-3 --epochs 20 --when 12 16 \
    --alpha 0 --beta 0 --wdecay 1.2e-6 --clip 0.25 \
    --wdrop 0.5 --dropoute 0 --dropouth 0.25 --dropouti 0.1 --dropout 0.1 \
    --save PTBC.pt

-----------------------------

Loading cached dataset...
Applying weight drop of 0.5 to weight_hh_l0
Applying weight drop of 0.5 to weight_hh_l0
Applying weight drop of 0.5 to weight_hh_l0
Using []
Model total parameters: 13787650
| epoch   1 |   200/  261 batches | lr 0.00200 | ms/batch 244.16 | loss  2.97 | ppl    19.49 | bpc    4.285
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 74.52s | valid loss  1.91 | valid ppl     6.76 | valid bpc    2.756
-----------------------------------------------------------------------------------------
Saving model (new best validation)
"""

import argparse
import hashlib
import math
import os
import time
from collections import Counter, namedtuple
from collections import defaultdict
import warnings

import numpy as np
import torch
import torch.nn as nn
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import MultiStepLR


# https://github.com/salesforce/awd-lstm-lm/issues/7
warnings.simplefilter("ignore", UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--train_bs', type=int, default=80, metavar='N',
                        help='train batch size')
    parser.add_argument('--valid_bs', type=int, default=10, metavar='N',
                        help='valid batch size')
    parser.add_argument('--test_bs', type=int, default=1, metavar='N',
                        help='test batch size')

    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to multiply the learning rate by gamma - accepts multiple')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')

    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')

    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')

    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--resume', type=str, default='',
                        help='path of model to resume')

    args = parser.parse_args()
    args.tied = True

    return args


""" Model & Criterion """


class SplitCrossEntropyLoss(nn.Module):
    r'''SplitCrossEntropyLoss calculates an approximate softmax'''

    def __init__(self, hidden_size, splits, verbose=False):
        # We assume splits is [0, split1, split2, N] where N >= |V|
        # For example, a vocab of 1000 words may have splits [0] + [100, 500] + [inf]
        super(SplitCrossEntropyLoss, self).__init__()
        self.hidden_size = hidden_size
        self.splits = [0] + splits + [100 * 1000000]
        self.nsplits = len(self.splits) - 1
        self.stats = defaultdict(list)
        self.verbose = verbose
        # Each of the splits that aren't in the head require a pretend token, we'll call them tombstones
        # The probability given to this tombstone is the probability of selecting an item from the represented split
        if self.nsplits > 1:
            self.tail_vectors = nn.Parameter(torch.zeros(self.nsplits - 1, hidden_size))
            self.tail_bias = nn.Parameter(torch.zeros(self.nsplits - 1))

    def logprob(self, weight, bias, hiddens, splits=None, softmaxed_head_res=None):
        # First we perform the first softmax on the head vocabulary and the tombstones
        if softmaxed_head_res is None:
            start, end = self.splits[0], self.splits[1]
            head_weight = None if end - start == 0 else weight[start:end]
            head_bias = None if end - start == 0 else bias[start:end]
            # We only add the tombstones if we have more than one split
            if self.nsplits > 1:
                head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
                head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])

            # Perform the softmax calculation for the word vectors in the head for all splits
            # We need to guard against empty splits as torch.cat does not like random lists
            head_res = torch.nn.functional.linear(hiddens, head_weight, bias=head_bias)
            softmaxed_head_res = torch.nn.functional.log_softmax(head_res, dim=-1)

        if splits is None:
            splits = list(range(self.nsplits))

        results = []
        # running_offset = 0
        for idx in splits:

            # For those targets in the head (idx == 0) we only need to return their loss
            if idx == 0:
                results.append(softmaxed_head_res[:, :-(self.nsplits - 1)])

            # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
            else:
                start, end = self.splits[idx], self.splits[idx + 1]
                tail_weight = weight[start:end]
                tail_bias = bias[start:end]

                # Calculate the softmax for the words in the tombstone
                tail_res = torch.nn.functional.linear(hiddens, tail_weight, bias=tail_bias)

                # Then we calculate p(tombstone) * p(word in tombstone)
                # Adding is equivalent to multiplication in log space
                head_entropy = (softmaxed_head_res[:, -idx]).contiguous()
                tail_entropy = torch.nn.functional.log_softmax(tail_res, dim=-1)
                results.append(head_entropy.view(-1, 1) + tail_entropy)

        if len(results) > 1:
            return torch.cat(results, dim=1)
        return results[0]

    def split_on_targets(self, hiddens, targets):
        # Split the targets into those in the head and in the tail
        split_targets = []
        split_hiddens = []

        # Determine to which split each element belongs (for each start split value, add 1 if equal or greater)
        # This method appears slower at least for WT-103 values for approx softmax
        # masks = [(targets >= self.splits[idx]).view(1, -1) for idx in range(1, self.nsplits)]
        # mask = torch.sum(torch.cat(masks, dim=0), dim=0)
        ###
        # This is equally fast for smaller splits as method below but scales linearly
        mask = None
        for idx in range(1, self.nsplits):
            partial_mask = targets >= self.splits[idx]
            mask = mask + partial_mask if mask is not None else partial_mask
        ###
        # masks = torch.stack([targets] * (self.nsplits - 1))
        # mask = torch.sum(masks >= self.split_starts, dim=0)
        for idx in range(self.nsplits):
            # If there are no splits, avoid costly masked select
            if self.nsplits == 1:
                split_targets, split_hiddens = [targets], [hiddens]
                continue
            # If all the words are covered by earlier targets, we have empties so later stages don't freak out
            if sum(len(t) for t in split_targets) == len(targets):
                split_targets.append([])
                split_hiddens.append([])
                continue
            # Are you in our split?
            tmp_mask = mask == idx
            split_targets.append(torch.masked_select(targets, tmp_mask))
            split_hiddens.append(
                hiddens.masked_select(tmp_mask.unsqueeze(1).expand_as(hiddens)).view(-1, hiddens.size(1)))
        return split_targets, split_hiddens

    def forward(self, weight, bias, hiddens, targets, verbose=False):
        if self.verbose or verbose:
            for idx in sorted(self.stats):
                print('{}: {}'.format(idx, int(np.mean(self.stats[idx]))), end=', ')
            print()

        total_loss = None
        if len(hiddens.size()) > 2: hiddens = hiddens.view(-1, hiddens.size(2))

        split_targets, split_hiddens = self.split_on_targets(hiddens, targets)

        # First we perform the first softmax on the head vocabulary and the tombstones
        start, end = self.splits[0], self.splits[1]
        head_weight = None if end - start == 0 else weight[start:end]
        head_bias = None if end - start == 0 else bias[start:end]

        # We only add the tombstones if we have more than one split
        if self.nsplits > 1:
            head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
            head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])

        # Perform the softmax calculation for the word vectors in the head for all splits
        # We need to guard against empty splits as torch.cat does not like random lists
        combo = torch.cat([split_hiddens[i] for i in range(self.nsplits) if len(split_hiddens[i])])
        ###
        all_head_res = torch.nn.functional.linear(combo, head_weight, bias=head_bias)
        softmaxed_all_head_res = torch.nn.functional.log_softmax(all_head_res, dim=-1)
        if self.verbose or verbose:
            self.stats[0].append(combo.size()[0] * head_weight.size()[0])

        running_offset = 0
        for idx in range(self.nsplits):
            # If there are no targets for this split, continue
            if len(split_targets[idx]) == 0: continue

            # For those targets in the head (idx == 0) we only need to return their loss
            if idx == 0:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]
                entropy = -torch.gather(softmaxed_head_res, dim=1, index=split_targets[idx].view(-1, 1))
            # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
            else:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]

                if self.verbose or verbose:
                    start, end = self.splits[idx], self.splits[idx + 1]
                    tail_weight = weight[start:end]
                    self.stats[idx].append(split_hiddens[idx].size()[0] * tail_weight.size()[0])

                # Calculate the softmax for the words in the tombstone
                tail_res = self.logprob(weight, bias, split_hiddens[idx], splits=[idx],
                                        softmaxed_head_res=softmaxed_head_res)

                # Then we calculate p(tombstone) * p(word in tombstone)
                # Adding is equivalent to multiplication in log space
                head_entropy = softmaxed_head_res[:, -idx]
                # All indices are shifted - if the first split handles [0,...,499] then the 500th in the second split will be 0 indexed
                indices = (split_targets[idx] - self.splits[idx]).view(-1, 1)
                # Warning: if you don't squeeze, you get an N x 1 return, which acts oddly with broadcasting
                tail_entropy = torch.gather(torch.nn.functional.log_softmax(tail_res, dim=-1), dim=1,
                                            index=indices).squeeze()
                entropy = -(head_entropy + tail_entropy)
            ###
            running_offset += len(split_hiddens[idx])
            total_loss = entropy.float().sum() if total_loss is None else total_loss + entropy.float().sum()

        return (total_loss / len(targets)).type_as(weight)


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    # noinspection PyProtectedMember
    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            new_param = nn.Parameter(getattr(self.module, name_w).data)
            delattr(self.module, name_w)
            setattr(self.module, name_w + '_raw', new_param)
            # https://github.com/salesforce/awd-lstm-lm/issues/2
            # https://github.com/pytorch/pytorch/blob/0c93/torch/nn/modules/rnn.py#L65
            if isinstance(self.module, torch.nn.RNNBase):
                for i, n in enumerate(self.module._flat_weights_names):
                    if n == name_w:
                        self.module._flat_weights_names[i] = name_w + '_raw'

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            if self.variational:
                mask = torch.ones(raw_w.size(0), 1, requires_grad=True)
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            # https://github.com/salesforce/awd-lstm-lm/issues/114
            setattr(self.module, name_w, torch.nn.Parameter(w))

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X

# NOTE: interface: init_hidden, decoder.weight/bias, dropoutx, rnns
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1,
                 wdrop=0, tie_weights=False):

        super(RNNModel, self).__init__()

        self.rnn_type = rnn_type

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0)
                         for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0)
                         for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        else:
            raise NotImplementedError(f'RNN type {rnn_type} is not supported')

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self._init_weights()
        self.tie_weights = tie_weights

    def _init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0) * output.size(1), output.size(2))
        if return_h: return result, hidden, raw_outputs, outputs
        else: return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                         self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors,
        to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


def _override_hparams(model, args):
    model.dropouti = args.dropouti
    model.dropouth = args.dropouth
    model.dropout = args.dropout
    model.dropoute = args.dropoute
    if not args.wdrop: return
    for rnn in model.rnns:
        if type(rnn) == WeightDrop:
            rnn.dropout = args.wdrop
        elif rnn.zoneout > 0:
            rnn.zoneout = args.wdrop


""" Dataset """


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class Dataset:
    def __init__(self, train, valid, test, train_bs, valid_bs, test_bs, ntokens):
        self.train = train
        self.valid = valid
        self.test = test
        self.train_bs = train_bs
        self.valid_bs = valid_bs
        self.test_bs = test_bs
        self.ntokens = ntokens

    @staticmethod
    def get_batch(source, i, seq_len):
        seq_len = min(seq_len, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target


""" Learner """


LearnerConfig = namedtuple('LearnerConfig', [
    'bptt', 'clip', 'alpha', 'beta', 'log_interval',
    'epochs', 'lr', 'save',
])


class Learner:

    def __init__(self, model, criterion, optimizer, scheduler, dataset: Dataset, config: LearnerConfig):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.config = config

    def train_end_to_end(self):
        best_loss = 1e8

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            self.train_one_epoch(epoch)
            self.scheduler.step()
            val_loss = self.validate(epoch, epoch_start_time)
            if val_loss < best_loss:
                self.model_save()
                print('Saving model (new best validation)')
                best_loss = val_loss

    def train_one_epoch(self, epoch):
        total_loss = 0
        start_time = time.time()
        hidden = self.model.init_hidden(self.dataset.train_bs)
        batch, i = 0, 0

        while i < self.dataset.train.size(0) - 1 - 1:
            bptt = self.config.bptt if np.random.random() < 0.95 else self.config.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)

            lr2 = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.config.bptt
            self.model.train()
            data, targets = Dataset.get_batch(self.dataset.train, i, seq_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = self.model.repackage_hidden(hidden)
            self.optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, return_h=True)
            raw_loss = self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets)

            loss = raw_loss

            if self.config.alpha:  # Activiation Regularization
                loss = loss + sum(self.config.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            if self.config.beta:  # Temporal Activation Regularization (slowness)
                loss = loss + sum(self.config.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
            params = list(self.model.parameters()) + list(self.criterion.parameters())
            if self.config.clip: torch.nn.utils.clip_grad_norm_(params, self.config.clip)
            self.optimizer.step()

            total_loss += raw_loss.data
            self.optimizer.param_groups[0]['lr'] = lr2
            if batch % self.config.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / self.config.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch, len(self.dataset.train) // self.config.bptt, self.optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / self.config.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2),
                ))
                total_loss = 0
                start_time = time.time()
            batch += 1
            i += seq_len

    def validate(self, epoch=None, epoch_start_time=None):
        epoch_start_time = epoch_start_time or time.time()
        val_loss = self._evaluate(self.dataset.valid, self.dataset.valid_bs)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
        print('-' * 89)
        return val_loss

    def test(self, save_path):  # args.save
        self.model_load(save_path)  # Load the best saved model
        test_loss = self._evaluate(self.dataset.test, self.dataset.test_bs)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('=' * 89)

    def _evaluate(self, data_source, batch_size=10):
        self.model.eval()  # Turn on evaluation mode which disables dropout.
        total_loss = 0
        hidden = self.model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, self.config.bptt):
            data, targets = Dataset.get_batch(data_source, i, self.config.bptt)
            output, hidden = self.model(data, hidden)
            total_loss += len(data) * self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets).data
            hidden = self.model.repackage_hidden(hidden)
        return total_loss.item() / len(data_source)

    def model_save(self, fn=None):
        with open(fn or self.config.save, 'wb') as f:
            torch.save([self.model, self.criterion, self.optimizer, self.scheduler], f)

    def model_load(self, fn):
        with open(fn, 'rb') as f:
            self.model, self.criterion, self.optimizer, self.scheduler = torch.load(f)


class LearnerFactory:

    @staticmethod
    def build_learner(args):
        LearnerFactory._set_seed(args)
        dataset = LearnerFactory._build_dataset(args)
        config = LearnerFactory._build_config(args)
        if args.resume:
            model, criterion, optimizer, scheduler = LearnerFactory._resume_from(args)
        else:
            model = LearnerFactory._build_model(args, dataset.ntokens)
            criterion = LearnerFactory._build_criterion(args, dataset.ntokens)
            optimizer = LearnerFactory._build_optimizer(args, model, criterion)
            scheduler = LearnerFactory._build_scheduler(args, optimizer)
        return Learner(model, criterion, optimizer, scheduler, dataset, config)

    @staticmethod
    def _build_scheduler(args, optimizer):
        return MultiStepLR(optimizer, milestones=args.when, gamma=args.gamma)

    @staticmethod
    def _build_dataset(args):

        def batchify(data, bsz, use_cuda):
            # Work out how cleanly we can divide the dataset into bsz parts.
            nbatch = data.size(0) // bsz
            # Trim off any extra elements that wouldn't cleanly fit (remainders).
            data = data.narrow(0, 0, nbatch * bsz)
            # Evenly divide the data across the bsz batches.
            data = data.view(bsz, -1).t().contiguous()
            return data.cuda() if use_cuda else data

        fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
        if os.path.exists(fn):
            print('Loading cached dataset...')
            corpus = torch.load(fn)
        else:
            print('Producing dataset...')
            corpus = Corpus(args.data)
            torch.save(corpus, fn)

        return Dataset(
            train=batchify(corpus.train, args.train_bs, args.cuda), train_bs=args.train_bs,
            valid=batchify(corpus.valid, args.valid_bs, args.cuda), valid_bs=args.valid_bs,
            test=batchify(corpus.test, args.test_bs, args.cuda), test_bs=args.test_bs,
            ntokens=len(corpus.dictionary),
        )

    @staticmethod
    def _build_config(args):
        return LearnerConfig(
            bptt=args.bptt, clip=args.clip,
            alpha=args.alpha, beta=args.beta, log_interval=args.log_interval,
            epochs=args.epochs, lr=args.lr, save=args.save,
        )

    @staticmethod
    def _set_seed(args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            if not args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(args.seed)

    @staticmethod
    def _build_criterion(args, ntokens):
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using', splits)
        criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
        return criterion.cuda() if args.cuda else criterion

    @staticmethod
    def _build_model(args, ntokens):
        model = RNNModel(
            args.model, ntokens, args.emsize, args.nhid, args.nlayers,
            args.dropout, args.dropouth, args.dropouti,
            args.dropoute, args.wdrop, args.tied
        )
        return model.cuda() if args.cuda else model

    @staticmethod
    def _build_optimizer(args, model, criterion):
        params = list(model.parameters()) + list(criterion.parameters())
        total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
        print('Model total parameters:', total_params)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
        else:
            raise NotImplementedError
        return optimizer

    @staticmethod
    def _resume_from(args):
        print('Resuming model ...')
        model, criterion, optimizer, scheduler = torch.load(open(args.resume, 'rb'))
        optimizer.param_groups[0]['lr'] = args.lr
        return model, criterion, optimizer, scheduler


if __name__ == "__main__":
    learner = LearnerFactory.build_learner(parse_args())
    learner.train_end_to_end()
