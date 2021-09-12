#!/usr/bin/env python
# -*-coding:utf8-*-
""" Translator Class and builder """
from __future__ import print_function
import codecs
import torch

from tensorboardX import SummaryWriter
from others.utils import test_bleu, test_length, test_rouge
from others.utils import rouge_results_to_str
from translate.beam import GNMTGlobalScorer
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def build_predictor(args, tokenizer, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')

    translator = Translator(args, model, tokenizer,
                            global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 tokenizer,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.start_token = self.vocab['[unused1]']
        self.end_token = self.vocab['[unused2]']
        self.seg_token = self.vocab['[unused3]']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(
            tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}


    def from_batch_dev(self, doc_batch, tgt_data):

        translations = []

        batch_size = len(doc_batch)

        for b in range(batch_size):

            # generated text
            pred_summ = self.tokenizer.convert_ids_to_tokens(
                [int(n) for n in doc_batch[b]])
            pred_summ = ' '.join(pred_summ)
            pred_summ = pred_summ.replace('[unused0]', '').replace('[unused1]', '').\
                replace('[unused2]', '').replace('[unused5]', '#').replace('[UNK]', '').replace(' ##', '').replace('[unused3]', '<q>').strip()
            pred_summ = ' '.join(pred_summ.split())

            gold_data = tgt_data[b]
            gold_data = self.tokenizer.convert_ids_to_tokens([int(n) for n in gold_data])
            gold_data = ' '.join(gold_data)
            gold_data = gold_data.replace('[PAD]', '').replace('[unused1]', '').\
                replace('[unused2]', '').replace('[unused5]', '#').replace("[UNK]", '').replace(' ##', '').replace('[unused3]', '<q>').strip()
            gold_data = ' '.join(gold_data.split())

            translations.append((pred_summ, gold_data))

        return translations

    def from_batch_test(self, batch, output_batch, tgt_data):

        translations = []

        batch_size = len(batch)

        origin_txt, ex_segs = batch.original_str, batch.ex_segs

        ex_segs = [sum(ex_segs[:i]) for i in range(len(ex_segs)+1)]

        for b in range(batch_size):
            # original text
            original_sent = ' <S> '.join(origin_txt[ex_segs[b]:ex_segs[b+1]])

            # long doc context text
            pred_summ = self.tokenizer.convert_ids_to_tokens(
                [int(n) for n in output_batch[b]])
            pred_summ = ' '.join(pred_summ)

            pred_summ = pred_summ.replace('[unused0]', '').replace('[unused1]', '').\
                replace('[unused2]', '').replace('[unused5]', '#').replace('[UNK]', '').replace(' ##', '').replace('[unused3]', '<q>').strip()
            pred_summ = ' '.join(pred_summ.split())

            gold_data = tgt_data[b]
            gold_data = self.tokenizer.convert_ids_to_tokens([int(n) for n in gold_data])
            gold_data = ' '.join(gold_data)
            gold_data = gold_data.replace('[PAD]', '').replace('[unused1]', '').replace('[unused2]', '').\
                replace('[unused5]', '#').replace('[UNK]', '').replace(' ##', '').replace('[unused3]', '<q>').strip()
            gold_data = ' '.join(gold_data.split())

            translation = (original_sent, pred_summ, gold_data)
            translations.append(translation)

        return translations

    def validate(self, data_iter, step, attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + 'step.%d.gold_temp' % step
        pred_path = self.args.result_path + 'step.%d.pred_temp' % step
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        pred_out_file = codecs.open(pred_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                output_data, tgt_data = self.translate_batch(batch)
                translations = self.from_batch_dev(output_data, tgt_data)

                for idx in range(len(translations)):
                    if ct % 100 == 0:
                        print("Processing %d" % ct)
                    pred_summ, gold_data = translations[idx]
                    pred_out_file.write(pred_summ + '\n')
                    gold_out_file.write(gold_data + '\n')
                    ct += 1
                pred_out_file.flush()
                gold_out_file.flush()

        pred_out_file.close()
        gold_out_file.close()

        if (step != -1):
            pred_bleu = test_bleu(pred_path, gold_path)
            gold_file = codecs.open(gold_path, 'r', 'utf-8')
            pred_file = codecs.open(pred_path, 'r', 'utf-8')
            pred_rouges = test_rouge(pred_file, gold_file)
            self.logger.info('Gold Length at step %d: %.2f' %
                             (step, test_length(gold_path, gold_path, ratio=False)))
            self.logger.info('Prediction Length ratio at step %d: %.2f' %
                             (step, test_length(pred_path, gold_path)))
            self.logger.info('Prediction Bleu at step %d: %.2f' %
                             (step, pred_bleu*100))
            self.logger.info('Prediction Rouges at step %d: \n%s\n' %
                             (step, rouge_results_to_str(pred_rouges)))
            rouge_results = (pred_rouges["rouge-1"]['f'],
                             pred_rouges["rouge-2"]['f'],
                             pred_rouges["rouge-l"]['f'])
            gold_file.close()
            pred_file.close()
        return rouge_results

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        output_path = self.args.result_path + '.%d.output' % step
        output_file = codecs.open(output_path, 'w', 'utf-8')
        gold_path = self.args.result_path + '.%d.gold_test' % step
        pred_path = self.args.result_path + '.%d.pred_test' % step
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        pred_out_file = codecs.open(pred_path, 'w', 'utf-8')
        ct = 0

        with torch.no_grad():
            rouge = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2, length_limit_type='words')
            for batch in data_iter:
                output_data, tgt_data = self.translate_batch(batch)
                translations = self.from_batch_test(batch, output_data, tgt_data)

                for idx in range(len(translations)):
                    origin_sent, pred_summ, gold_data = translations[idx]
                    if ct % 100 == 0:
                        print("Processing %d" % ct)
                    output_file.write("ID      : %d\n" % ct)
                    output_file.write("ORIGIN  : \n    " + origin_sent.replace('<S>', '\n    ') + "\n")
                    output_file.write("GOLD    : " + gold_data + "\n")
                    output_file.write("DOC_GEN : " + pred_summ + "\n")
                    rouge_score = rouge.get_scores([pred_summ], [gold_data])
                    bleu_score = sentence_bleu([gold_data.split()], pred_summ.split(),
                                               smoothing_function=SmoothingFunction().method1)
                    output_file.write("DOC_GEN  bleu & rouge-f 1/l:    %.4f & %.4f/%.4f\n\n" %
                                      (bleu_score, rouge_score["rouge-1"]["f"], rouge_score["rouge-l"]["f"]))
                    pred_out_file.write(pred_summ.strip() + '\n')
                    gold_out_file.write(gold_data.strip() + '\n')
                    ct += 1
                pred_out_file.flush()
                gold_out_file.flush()
                output_file.flush()

        pred_out_file.close()
        gold_out_file.close()
        output_file.close()

        if (step != -1):
            pred_bleu = test_bleu(pred_path, gold_path)
            gold_file = codecs.open(gold_path, 'r', 'utf-8')
            pred_file = codecs.open(pred_path, 'r', 'utf-8')
            pred_rouges = test_rouge(pred_file, gold_file)
            self.logger.info('Gold Length at step %d: %.2f\n' %
                             (step, test_length(gold_path, gold_path, ratio=False)))
            self.logger.info('Prediction Length ratio at step %d: %.2f' %
                             (step, test_length(pred_path, gold_path)))
            self.logger.info('Prediction Bleu at step %d: %.2f' %
                             (step, pred_bleu*100))
            self.logger.info('Prediction Rouges at step %d: \n%s' %
                             (step, rouge_results_to_str(pred_rouges)))
            gold_file.close()
            pred_file.close()

    def translate_batch(self, batch):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        # model output is (doc_decode_output, recon_results, summary_results)
        _, output_data = self.model(batch)
        dec_tgt = batch.dec_tgt
        return output_data, dec_tgt


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
