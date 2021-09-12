# -*- coding:utf-8 -*-

import gc
import glob
import json
import os
import re
import torch
import random
from os.path import join as pjoin

from multiprocess import Pool
from others.logging import logger
from transformers import BertTokenizer


dialog_dataset = ['reddit']
summ_dataset = ['cnndm', 'gigaword', 'newsroom']
doc_dataset = ['bookcorpus', 'visdial']
dialog_summ_dataset = ['samsum', 'adsc']


def clean(sent):
    needed_char = '''a-zA-Z0-9,.!:'‘’()?\-\$\"'''
    ret = []
    for word in sent:
        cleaned_word = ''.join([char for char in word if re.match("^[" + needed_char + "]*$", char)])
        if len(cleaned_word) > 0:
            ret.append(cleaned_word)
    return ret


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        self.tgt_seg = '[unused3]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.unk_vid = self.tokenizer.vocab[self.unk_token]

    def preprocess_src(self, content):
        if_exceed_length = False

        if len(content) < self.args.min_src_ntokens_per_sent:
            return None
        if len(content) > self.args.max_src_ntokens_per_sent:
            if_exceed_length = True

        original_txt = ' '.join(content)

        if self.args.truncated:
            content = content[:self.args.max_src_ntokens_per_sent]
        content_text = ' '.join(content)
        content_subtokens = self.tokenizer.tokenize(content_text)

        # [CLS] + T0 + T1 + ... + Tn
        src_subtokens = [self.cls_token] + content_subtokens
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        segments_ids = len(src_subtoken_idxs) * [0]

        return src_subtoken_idxs, segments_ids, original_txt, \
            src_subtokens, if_exceed_length

    def preprocess_summary(self, content):

        content = [' '.join(sent) for sent in content]
        content_subtokens_str = (' ' + self.tgt_seg + ' ').join(
            [' '.join(self.tokenizer.tokenize(sent)) for sent in content])
        content_subtokens = content_subtokens_str.split()[:self.args.max_tgt_len]
        original_txt = ' '.join(content_subtokens).replace(self.tgt_seg, '<q>').replace(' ##', '')
        content_subtokens = [self.tgt_bos] + content_subtokens + [self.tgt_eos]
        subtoken_idxs = self.tokenizer.convert_tokens_to_ids(content_subtokens)

        return subtoken_idxs, original_txt, content_subtokens

    def integrate_doc(self, doc):
        src_tokens = [self.cls_token]
        segments_ids = [0]
        segment_id = 0
        for sent in doc:
            tokens = sent["src_tokens"][1:] + [self.sep_token]
            src_tokens.extend(tokens)
            segments_ids.extend([segment_id] * len(tokens))
            segment_id = 1 - segment_id
        src_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
        return {"src_id": src_ids, "segs": segments_ids}


def format_json_to_data(args, corpus_type='train'):

    a_lst = []
    if args.mix_up:
        corpus_list = dialog_dataset + summ_dataset + doc_dataset
        if corpus_type is not None:
            for corpus_name in corpus_list:
                raw_path = pjoin(args.raw_path, corpus_name)
                for json_f in glob.glob(pjoin(raw_path, '*' + corpus_type + '*.json')):
                    real_name = json_f.split('/')[-1]
                    a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'pt'))))
        else:
            for corpus_name in corpus_list:
                raw_path = pjoin(args.raw_path, corpus_name)
                for json_f in glob.glob(pjoin(raw_path, '*.json')):
                    real_name = json_f.split('/')[-1]
                    corpus_type = real_name.split('.')[1]
                    a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'pt'))))
    else:
        if corpus_type is not None:
            for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '*.json')):
                real_name = json_f.split('/')[-1]
                a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'pt'))))
        else:
            for json_f in glob.glob(pjoin(args.raw_path, '*.json')):
                real_name = json_f.split('/')[-1]
                corpus_type = real_name.split('.')[1]
                a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'pt'))))

    total_statistic = {
        "instances": 0,
        "total_sents": 0.,
        "processed_sents": 0.,
        "max_sents": -1,
        "sents_num": [0] * 11,
        "exceed_length_num": 0,
        "exceed_sents_num": 0,
        "total_src_length": 0.,
        "src_sent_length_num": [0] * 11,
        "src_token_length_num": [0] * 11,
        "total_tgt_length": 0
    }

    pool = Pool(args.n_cpus)
    for statistic in pool.imap(_format_json_to_data, a_lst):
        if statistic is None:
            continue
        total_statistic["instances"] += statistic["instances"]
        total_statistic["total_sents"] += statistic["total_sents"]
        total_statistic["processed_sents"] += statistic["processed_sents"]
        total_statistic["max_sents"] = max(total_statistic["max_sents"], statistic["max_sents"])
        total_statistic["exceed_length_num"] += statistic["exceed_length_num"]
        total_statistic["exceed_sents_num"] += statistic["exceed_sents_num"]
        total_statistic["total_src_length"] += statistic["total_src_length"]
        total_statistic["total_tgt_length"] += statistic["total_tgt_length"]
        for idx in range(len(total_statistic["sents_num"])):
            total_statistic["sents_num"][idx] += statistic["sents_num"][idx]
        for idx in range(len(total_statistic["src_sent_length_num"])):
            total_statistic["src_sent_length_num"][idx] += statistic["src_sent_length_num"][idx]
        for idx in range(len(total_statistic["src_token_length_num"])):
            total_statistic["src_token_length_num"][idx] += statistic["src_token_length_num"][idx]

    pool.close()
    pool.join()

    if total_statistic["instances"] > 0:
        logger.info("Total examples: %d" %
                    total_statistic["instances"])
        logger.info("Average sentence number per document: %f" %
                    (total_statistic["total_sents"] / total_statistic["instances"]))
        logger.info("Processed average sentence number per document: %f" %
                    (total_statistic["processed_sents"] / total_statistic["instances"]))
        logger.info("Total sentences: %d" %
                    total_statistic["total_sents"])
        logger.info("Processed sentences: %d" %
                    total_statistic["processed_sents"])
        logger.info("Exceeded max sentence number documents: %d" %
                    total_statistic["exceed_sents_num"])
        logger.info("Max document sentences: %d" %
                    total_statistic["max_sents"])
        for idx, num in enumerate(total_statistic["sents_num"]):
            logger.info("document sentences %d ~ %d: %d, %.2f%%" %
                        (idx * 20, (idx+1) * 20, num, (num / total_statistic["instances"])))
        logger.info("Exceed length sentences number: %d" %
                    total_statistic["exceed_length_num"])
        logger.info("Average src sentence length: %f" %
                    (total_statistic["total_src_length"] / total_statistic["total_sents"]))
        for idx, num in enumerate(total_statistic["src_sent_length_num"]):
            logger.info("Sent length %d ~ %d: %d, %.2f%%" %
                        (idx * 10, (idx+1) * 10, num, (num / total_statistic["total_sents"])))
        logger.info("Average src token length: %f" %
                    (total_statistic["total_src_length"] / total_statistic["instances"]))
        for idx, num in enumerate(total_statistic["src_token_length_num"]):
            logger.info("token num %d ~ %d: %d, %.2f%%" %
                        (idx * 300, (idx+1) * 300, num, (num / total_statistic["instances"])))
        logger.info("Average tgt length: %f" %
                    (total_statistic["total_tgt_length"] / total_statistic["instances"]))

    if args.mix_up:
        logger.info("Mixing up all datasets...")
        file_count = len([0 for _ in glob.glob(pjoin(args.save_path, '*.pt'))])
        ex_count = 0
        random.seed(args.random_seed)
        file_set = {i: args.shard_size for i in range(file_count)}
        for i, pt_f in enumerate(glob.glob(pjoin(args.save_path, '*.pt'))):
            dataset = torch.load(pt_f)
            logger.info("Step: %d: mixing up %s" % (i, pt_f))
            ex_count += len(dataset)
            for ex in dataset:
                tmp = -1
                while tmp not in file_set.keys():
                    tmp = random.randint(0, file_count-1)
                with open(pjoin(args.save_path, str(tmp)+'.tmp.json'), 'a', encoding="utf-8") as save:
                    line = json.dumps(ex)
                    save.write(line+'\n')
                file_set[tmp] -= 1
                if file_set[tmp] <= 0:
                    del file_set[tmp]
            os.remove(pt_f)
        logger.info("Total files: %d" % file_count)
        logger.info("Total examples: %d" % ex_count)

        for i, f_name in enumerate(glob.glob(pjoin(args.save_path, '*.json'))):
            dataset = []
            with open(f_name, 'r', encoding='utf8') as json_f:
                for line in json_f:
                    ex = json.loads(line)
                    dataset.append(ex)
            random.shuffle(dataset)
            save_path = pjoin(args.save_path, 'data.train.'+str(i)+'.pt')
            torch.save(dataset, save_path)
            logger.info("Saving to %s with %d samples." % (save_path, len(dataset)))
            os.remove(f_name)
        gc.collect()


def _format_json_to_data(params):
    _, json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    datasets = []
    exceed_length_num = 0
    exceed_sents_num = 0
    total_src_length = 0.
    total_tgt_length = 0.
    src_length_sent_num = [0] * 11
    src_length_token_num = [0] * 11
    max_sents = 0
    sents_num = [0] * 11
    document_sents = 0.
    processed_sents = 0.

    for document in jobs:
        document_b_data = []
        document_token_num = 0
        for index, sent in enumerate(document['src']):
            sent = clean(sent)
            b_data = bert.preprocess_src(sent)
            if (b_data is None):
                continue
            src_subtoken_idxs, segments_ids, original_txt, \
                src_subtokens, exceed_length = b_data

            b_data_dict = {"index": index,
                           "src_id": src_subtoken_idxs,
                           "segs": segments_ids,
                           "original_txt": original_txt,
                           "src_tokens": src_subtokens}

            src_length_sent_num[min(len(src_subtoken_idxs) // 10, 10)] += 1
            document_token_num += len(src_subtoken_idxs)
            total_src_length += len(src_subtoken_idxs)
            document_b_data.append(b_data_dict)
            if exceed_length:
                exceed_length_num += 1
            if len(document_b_data) >= args.max_sents:
                exceed_sents_num += 1
                if args.truncated:
                    break
        document_example = {"session": document_b_data}
        document_integrated = bert.integrate_doc(document_b_data)
        document_example["document"] = document_integrated
        if document['domain'] in dialog_dataset:
            document_example['domain'] = 0
        elif document['domain'] in doc_dataset:
            document_example['domain'] = 1
        elif document['domain'] in summ_dataset or \
                document['domain'] in dialog_summ_dataset:
            document_example['domain'] = 2
        else:
            raise Exception(
                "An input sample with the domain %s is not in domain candidates." %
                (document['domain']))
        # summary data process
        if 'tgt' in document:
            cleaned_sents = [clean(sent) for sent in document['tgt']]
            summ_b_data = bert.preprocess_summary(cleaned_sents)
            subtoken_idxs, original_txt, content_subtokens = summ_b_data
            total_tgt_length += len(subtoken_idxs)
            b_data_dict = {"id": subtoken_idxs,
                           "original_txt": original_txt,
                           "content_tokens": content_subtokens}
            document_example["summary"] = b_data_dict

        if len(document_b_data) >= args.min_sents:
            datasets.append(document_example)
            sents_num[min(len(document_b_data) // 20, 10)] += 1
            src_length_token_num[min(document_token_num // 300, 10)] += 1
            max_sents = max(max_sents, len(document_b_data))
            document_sents += len(document['src'])
            processed_sents += len(document_b_data)

    statistic = {
        "instances": len(datasets),
        "total_sents": document_sents,
        "processed_sents": processed_sents,
        "max_sents": max_sents,
        "sents_num": sents_num,
        "exceed_length_num": exceed_length_num,
        "exceed_sents_num": exceed_sents_num,
        "total_src_length": total_src_length,
        "src_sent_length_num": src_length_sent_num,
        "src_token_length_num": src_length_token_num,
        "total_tgt_length": total_tgt_length
    }

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
    return statistic
