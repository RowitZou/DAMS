#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic args
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-test_mode", default='abs', type=str, choices=['ext', 'abs'])
    parser.add_argument("-src_data_mode", default='sent', type=str, choices=['sent', 'word'])
    parser.add_argument("-data_path", default='torch_data')
    parser.add_argument("-model_path", default='models')
    parser.add_argument("-result_path", default='results/samsum')
    parser.add_argument("-bert_dir", default='bert/bert_base_uncased')
    parser.add_argument('-log_file', default='logs/temp.log')
    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-seed', default=666, type=int)
    parser.add_argument('-port', default=10000, type=int)

    # Batch sizes
    parser.add_argument("-batch_size", default=4000, type=int)
    parser.add_argument("-batch_ex_size", default=8, type=int)
    parser.add_argument("-test_batch_size", default=20000, type=int)
    parser.add_argument("-test_batch_ex_size", default=50, type=int)

    # Model args
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'transformer', 'rnn'])
    parser.add_argument("-decoder", default='transformer', type=str, choices=['transformer', 'rnn'])
    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=768, type=int)
    parser.add_argument("-enc_ff_size", default=2048, type=int)
    parser.add_argument("-enc_heads", default=8, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # args for copy mechanism and coverage
    parser.add_argument("-coverage", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-copy_attn", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-copy_attn_force", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-copy_loss_by_seqlength", type=str2bool, nargs='?', const=True, default=False)

    # args for sent-level encoder
    parser.add_argument("-hier_dropout", default=0.2, type=float)
    parser.add_argument("-hier_layers", default=3, type=int)
    parser.add_argument("-hier_hidden_size", default=768, type=int)
    parser.add_argument("-hier_heads", default=8, type=int)
    parser.add_argument("-hier_ff_size", default=2048, type=int)

    # Training process args
    parser.add_argument("-save_checkpoint_steps", default=2000, type=int)
    parser.add_argument("-accum_count", default=4, type=int)
    parser.add_argument("-report_every", default=3, type=int)
    parser.add_argument("-train_steps", default=200000, type=int)
    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-max_tgt_len", default=100, type=int)
    parser.add_argument("-train_enc_recon", default=False, nargs='?', const=True, type=str2bool)

    # Beam search decoding args
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=3, type=int)
    parser.add_argument("-min_length", default=2, type=int)
    parser.add_argument("-max_length", default=100, type=int)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    # Optim args
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-lr_bert", default=0.001, type=float)
    parser.add_argument("-lr_other", default=0.01, type=float)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-warmup_steps", default=10000, type=int)
    parser.add_argument("-warmup_steps_bert", default=10000, type=int)
    parser.add_argument("-warmup_steps_other", default=10000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    # Pretrain args
    parser.add_argument("-pretrain", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-pretrain_steps", default=250000, type=int)
    parser.add_argument("-adversarial", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-adv_start_step", default=0, type=int)
    parser.add_argument("-adv_lambda", default=0.1, type=float)
    parser.add_argument("-mask_token_prob", default=0.15, type=float)
    parser.add_argument("-select_sent_prob", default=0.80, type=float)

    # Utility args
    parser.add_argument("-val_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-val_start_from", default=-1, type=int)
    parser.add_argument("-train_from", default='')
    parser.add_argument("-train_from_ignore_optim", type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    from train import train
    from test import test
    from validate import validate

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.mode == 'train'):
        train(args, device_id)
    elif (args.mode == 'validate'):
        validate(args, device_id)
    elif (args.mode == 'test'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except RuntimeWarning:
            print("Unrecognized cp step.")
            step = 0
        test(args, device_id, cp, step)
    else:
        print("Undefined mode! Please check input.")
