import copy
import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

from models.decoder_tf import TransformerDecoder
from models.encoder import Bert, TransformerEncoder
from models.generator import Generator
from others.utils import tile
from models.neural import GradientReversal


class Model(nn.Module):
    def __init__(self, args, device, vocab, checkpoint=None):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.beam_size = args.beam_size
        self.max_length = args.max_length
        self.min_length = args.min_length

        # special tokens
        self.start_token = vocab['[unused1]']
        self.end_token = vocab['[unused2]']
        self.pad_token = vocab['[PAD]']
        self.mask_token = vocab['[MASK]']
        self.seg_token = vocab['[SEP]']
        self.cls_token = vocab['[CLS]']
        self.tgt_seg_token = vocab['[unused3]']

        if args.encoder == 'bert':
            self.encoder = Bert(args.bert_dir, args.finetune_bert)
            if(args.max_pos > 512):
                my_pos_embeddings = nn.Embedding(args.max_pos, self.encoder.model.config.hidden_size)
                my_pos_embeddings.weight.data[:512] = self.encoder.model.embeddings.position_embeddings.weight.data
                my_pos_embeddings.weight.data[512:] = self.encoder.model.embeddings.position_embeddings.\
                    weight.data[-1][None, :].repeat(args.max_pos-512, 1)
                self.encoder.model.embeddings.position_embeddings = my_pos_embeddings
            self.hidden_size = self.encoder.model.config.hidden_size
        else:
            self.hidden_size = args.enc_hidden_size
            self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
            self.encoder = TransformerEncoder(self.hidden_size, args.enc_ff_size, args.enc_heads,
                                              args.enc_dropout, args.enc_layers)

        tgt_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.hier = TransformerEncoder(self.hidden_size, args.hier_ff_size, args.hier_heads,
                                       args.hier_dropout, args.hier_layers)

        self.sent_decoder = TransformerDecoder(
            self.args.dec_layers, self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout,
            embeddings=tgt_embeddings)

        self.doc_decoder = TransformerDecoder(
            self.args.dec_layers, self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout,
            embeddings=tgt_embeddings)

        self.generator = Generator(self.vocab_size, self.args.dec_hidden_size, self.pad_token)

        # 2 discriminators
        if self.args.adversarial:
            self.discriminator1 = nn.Sequential(GradientReversal(),
                                                nn.Linear(self.hidden_size, self.hidden_size),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.hidden_size, 1))
            self.discriminator2 = nn.Sequential(GradientReversal(),
                                                nn.Linear(self.hidden_size, self.hidden_size),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.hidden_size, 1))

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
            self.doc_decoder.embeddings = self.sent_decoder.embeddings
            if args.share_emb:
                self.generator.linear.weight = self.sent_decoder.embeddings.weight
        else:
            # initialize params.
            if args.encoder == "transformer":
                for module in self.encoder.modules():
                    self._set_parameter_tf(module)
            for module in self.sent_decoder.modules():
                self._set_parameter_tf(module)
            for module in self.doc_decoder.modules():
                self._set_parameter_tf(module)
            for module in self.hier.modules():
                self._set_parameter_tf(module)
            for p in self.generator.parameters():
                self._set_parameter_linear(p)
            if self.args.adversarial:
                for p in self.discriminator1.parameters():
                    self._set_parameter_linear(p)
                for p in self.discriminator2.parameters():
                    self._set_parameter_linear(p)
            if args.share_emb:
                if args.encoder == 'bert':
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
                    tgt_embeddings.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
                    self.sent_decoder.embeddings = tgt_embeddings
                    self.doc_decoder.embeddings = tgt_embeddings
                self.generator.linear.weight = self.sent_decoder.embeddings.weight

        self.to(device)

    def _set_parameter_tf(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_parameter_linear(self, p):
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()

    def _add_mask(self, src, mask_src):
        pm_index = torch.empty_like(mask_src).float().uniform_().le(self.args.mask_token_prob)
        ps_index = torch.empty_like(mask_src[:, 0]).float().uniform_().gt(self.args.select_sent_prob)
        pm_index[ps_index] = 0
        # Avoid mask [PAD]
        pm_index[~mask_src] = 0
        # Avoid mask [CLS]
        pm_index[:, 0] = 0
        # Avoid mask [SEG]
        pm_index[src == self.seg_token] = 0
        src[pm_index] = self.mask_token
        return src

    def _fast_translate_batch(self, src, memory_bank, max_length, init_tokens=None,
                              memory_mask=None, min_length=2, beam_size=3, ignore_memory_attn=False, decode_type=None):

        batch_size = memory_bank.size(0)
        if decode_type == 'summary':
            dec_states = self.doc_decoder.init_decoder_state(src, memory_bank, with_cache=True)
        else:
            dec_states = self.sent_decoder.init_decoder_state(src, memory_bank, with_cache=True)
        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        memory_bank = tile(memory_bank, beam_size, dim=0)
        init_tokens = tile(init_tokens, beam_size, dim=0)
        memory_mask = tile(memory_mask, beam_size, dim=0)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=self.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=self.device)

        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=self.device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=self.device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = [[] for _ in range(batch_size)]  # noqa: F812

        for step in range(max_length):
            if step > 0:
                init_tokens = None
            # Decoder forward.
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)

            if decode_type == 'summary':
                dec_out, dec_states, _ = self.doc_decoder(decoder_input, memory_bank, dec_states, init_tokens, step=step,
                                                          memory_masks=memory_mask, ignore_memory_attn=ignore_memory_attn)
            else:
                dec_out, dec_states, _ = self.sent_decoder(decoder_input, memory_bank, dec_states, init_tokens, step=step,
                                                           memory_masks=memory_mask, ignore_memory_attn=ignore_memory_attn)

            # Generator forward.
            log_probs = self.generator(dec_out.transpose(0, 1).squeeze(0))

            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if(cur_len > 3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        if(len(words) <= 3):
                            continue
                        trigrams = [(words[i-1], words[i], words[i+1]) for i in range(1, len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            log_probs[i] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.args.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.floor_divide(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = torch.nonzero(is_finished[i]).view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        _, pred = best_hyp[0]
                        results[b].append(pred)
                non_finished = torch.nonzero(end_condition.eq(0)).view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            if memory_bank is not None:
                memory_bank = memory_bank.index_select(0, select_indices)
            if memory_mask is not None:
                memory_mask = memory_mask.index_select(0, select_indices)
            if init_tokens is not None:
                init_tokens = init_tokens.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        results = [t[0] for t in results]
        return results

    def pretrain(self, batch):

        # sent means sentence-level samples; word means word-level samples.
        ex_domains = batch.domains
        src = batch.sent_src
        mask_src = batch.mask_sent_src
        segs = batch.sent_segs
        enc_tgt = batch.enc_tgt
        dec_tgt = batch.dec_tgt
        ex_segs = batch.ex_segs

        # domain list for sentence-level input
        sent_domains = sum([[domain] * ex_segs[idx] for idx, domain in enumerate(ex_domains)], [])

        # construct filtered src and tgt
        filter_enc = torch.tensor(sent_domains, device=self.device).ne(1)
        filter_dec_src = torch.tensor(sent_domains, device=self.device).ne(0)
        filter_dec_tgt = torch.tensor(ex_domains, device=self.device).ne(0)
        enc_tgt = enc_tgt[filter_enc]
        dec_tgt = dec_tgt[filter_dec_tgt]
        adv_enc_tgt = torch.tensor([1. if domain == 2 else 0. for domain in sent_domains if domain != 1]).to(self.device)
        adv_dec_tgt = torch.tensor([1. if domain == 2 else 0. for domain in sent_domains if domain != 0]).to(self.device)
        dec_tgt2 = torch.tensor([1 if domain == 2 else 0 for domain in ex_domains if domain != 0]).to(self.device)
        setattr(batch, 'enc_tgt', enc_tgt)
        setattr(batch, 'adv_enc_tgt', adv_enc_tgt)
        setattr(batch, 'dec_tgt', dec_tgt)
        setattr(batch, 'adv_dec_tgt', adv_dec_tgt)
        setattr(batch, 'dec_tgt2', dec_tgt2)

        # encode src
        if self.training:
            src = self._add_mask(src, mask_src)
        if self.args.encoder == "bert":
            enc_hid = self.encoder(src, segs, mask_src)
        else:
            src_emb = self.embeddings(src)
            enc_hid = self.encoder(src_emb, ~mask_src)
        clss = enc_hid[:, 0, :]  # (sent_num, hidden_size)

        clss_enc = clss[filter_enc]
        clss_dec = clss[filter_dec_src]

        # enc_adv & sent decoder
        enc_adv_pred, sent_decode_output, recon_results = None, None, None
        if clss_enc.size(0) != 0:
            # adversarial of domain 0,2
            if self.args.adversarial:
                enc_adv_pred = torch.sigmoid(self.discriminator1(clss_enc)).squeeze(1)  # (sent_num)

            # decode and reconstruct src of domain 0,2
            if self.training:
                sent_dec_state = self.sent_decoder.init_decoder_state(src[filter_enc], clss_enc)
                sent_decode_output, _, _ = self.sent_decoder(enc_tgt[:, :-1], clss_enc, sent_dec_state,
                                                             ignore_memory_attn=True)
            else:
                recon_results = self._fast_translate_batch(src[filter_enc], clss_enc, self.max_length,
                                                           min_length=self.min_length, beam_size=self.beam_size,
                                                           ignore_memory_attn=True, decode_type='sent')

        doc_decode_output, dec_adv_pred, summary_results = None, None, None
        if clss_dec.size(0) != 0:
            # hier module
            ex_segs_dec = [ex_seg for idx, ex_seg in enumerate(ex_segs) if ex_domains[idx] != 0]
            clss_dec_list = torch.split(clss_dec, ex_segs_dec)
            hier_input = pad_sequence(clss_dec_list, batch_first=True, padding_value=0.)
            hier_mask_list = [mask_src.new_zeros([length]) for length in ex_segs_dec]
            hier_mask = pad_sequence(hier_mask_list, batch_first=True, padding_value=1)

            hier_output = self.hier(hier_input, hier_mask)

            # adversarial of domain 1,2
            if self.args.adversarial:
                hier_output_cat = hier_output.view(-1, hier_output.size(-1))[~hier_mask.view(-1)]
                dec_adv_pred = torch.sigmoid(self.discriminator2(hier_output_cat)).squeeze(1)

            # decode and generate tgt of domain 1,2
            if self.training:
                doc_dec_state = self.doc_decoder.init_decoder_state(batch.doc_src[filter_dec_tgt], hier_output)
                doc_decode_output, _, _ = self.doc_decoder(dec_tgt[:, :-1], hier_output, doc_dec_state,
                                                           memory_masks=hier_mask)
            else:
                summary_results = self._fast_translate_batch(batch.doc_src[filter_dec_tgt], hier_output, self.max_length,
                                                             min_length=self.min_length, beam_size=self.beam_size,
                                                             decode_type='summary', memory_mask=hier_mask)

        return sent_decode_output, doc_decode_output, recon_results, summary_results, enc_adv_pred, dec_adv_pred

    def forward(self, batch):

        # sent means sentence-level samples; word means word-level samples.
        src = batch.sent_src
        mask_src = batch.mask_sent_src
        segs = batch.sent_segs
        enc_tgt = batch.enc_tgt
        dec_tgt = batch.dec_tgt
        ex_segs = batch.ex_segs

        # encode src
        if self.args.encoder == "bert":
            enc_hid = self.encoder(src, segs, mask_src)
        else:
            src_emb = self.embeddings(src)
            enc_hid = self.encoder(src_emb, ~mask_src)
        clss = enc_hid[:, 0, :]  # (sent_num, hidden_size)

        # summary generate
        doc_decode_output, summary_results = None, None

        # hier module
        clss_dec_list = torch.split(clss, ex_segs)
        hier_input = pad_sequence(clss_dec_list, batch_first=True, padding_value=0.)
        hier_mask_list = [mask_src.new_zeros([length]) for length in ex_segs]
        hier_mask = pad_sequence(hier_mask_list, batch_first=True, padding_value=1)

        hier_output = self.hier(hier_input, hier_mask)

        if self.training:
            doc_dec_state = self.doc_decoder.init_decoder_state(batch.doc_src, hier_output)
            doc_decode_output, _, _ = self.doc_decoder(dec_tgt[:, :-1], hier_output, doc_dec_state,
                                                       memory_masks=hier_mask)
        else:
            summary_results = self._fast_translate_batch(batch.doc_src, hier_output, self.max_length,
                                                         min_length=self.min_length, beam_size=self.beam_size,
                                                         decode_type='summary', memory_mask=hier_mask)

        return doc_decode_output, summary_results
