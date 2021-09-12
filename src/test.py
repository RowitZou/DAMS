#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import torch

from transformers import BertTokenizer
from others.logging import logger
from models import data_loader
from models.data_loader import load_dataset
from models.dams import Model as Summarizer
from models.dams_predictor import build_predictor

model_flags = ['encoder', 'decoder', 'enc_heads', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_heads', 'dec_layers', 'dec_hidden_size', 'dec_ff_size']


def test(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)
    model.eval()
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, args.test_batch_ex_size, device,
                                       shuffle=False, is_test=True)

    predictor = build_predictor(args, tokenizer, model, logger)
    predictor.translate(test_iter, step)
