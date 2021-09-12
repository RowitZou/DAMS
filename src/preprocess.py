# encoding=utf-8

import argparse
from others.logging import init_logger
from prepro import json_to_data as data_builder


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", default='train', type=str, choices=['train', 'dev', 'test'])
    parser.add_argument("-raw_path", default='json_data', type=str)
    parser.add_argument("-save_path", default='torch_data/all', type=str)
    parser.add_argument("-n_cpus", default=1, type=int)
    parser.add_argument("-random_seed", default=666, type=int)

    # json_to_data args
    parser.add_argument('-log_file', default='logs/json_to_data.log')
    parser.add_argument("-bert_dir", default='bert/bert_base_uncased')
    parser.add_argument('-min_src_ntokens_per_sent', default=2, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=50, type=int)
    parser.add_argument('-min_sents', default=1, type=int)
    parser.add_argument('-max_sents', default=40, type=int)
    parser.add_argument('-max_tgt_len', default=100, type=int)
    parser.add_argument("-truncated", nargs='?', const=True, default=False)
    parser.add_argument("-mix_up", nargs='?', const=True, default=False)
    parser.add_argument("-shard_size", default=5000, type=int)

    args = parser.parse_args()
    init_logger(args.log_file)

    data_builder.format_json_to_data(args, args.type)
