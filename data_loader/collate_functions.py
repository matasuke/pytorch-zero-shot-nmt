from typing import List, Tuple

import torch


def seq2seq_collate_fn(
    inputs: List[Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, ...]:
    '''
    create mini-batch tensors from source target sentences.
    use this collate_fn to pad sentences.

    :param inputs: mini batch of source and target sentences with languages and styles.
    '''
    def merge_seq(sentences: List[torch.Tensor]):
        '''
        pad sequences for source
        '''
        lengths = [len(sen) for sen in sentences]
        padded_seqs = torch.zeros(len(sentences), max(lengths)).long()

        for idx, sen in enumerate(sentences):
            end = lengths[idx]
            padded_seqs[idx, :end] = sen[:end]

        padded_seqs = padded_seqs.t().contiguous()

        return padded_seqs, lengths

    indices = list(range(len(inputs)))

    # sort a list of sentence length based on source sentence to use pad_padded_sequence
    src_tgt_pair, indices = \
        zip(*sorted(zip(inputs, indices), key=lambda x: len(x[0][0]), reverse=True))
    src, tgt = zip(*src_tgt_pair)

    src, lengths = merge_seq(src)
    tgt, _ = merge_seq(tgt)

    lengths = torch.LongTensor([lengths])

    return src, tgt, lengths, indices
