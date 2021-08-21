import json
import numpy as np
import os
import pandas as pd
import re
import torch
import spacy
import traceback
import transformers
from cache_decorator import Cache
from collections import defaultdict
from tqdm.notebook import tqdm
from transformers import (
    BartTokenizer, BertTokenizer
)
from typing import List, Tuple, Union

spacy.prefer_gpu()

"""
@Cache(
    cache_path=[
        "cache/h_events/{function_name}/{_hash}/tokens_results_{_hash}.pkl",
        "cache/h_events/{function_name}/{_hash}/tags_results_{_hash}.pkl"
    ],
    args_to_ignore=["paragraph_dict", "nlp"]
)
"""
def tag_paragraph(
    paragraph_dict: dict,
    nlp: spacy.language.Language,
    par_title: str,
    par_num: int
):
    # Get the tokens.
    text = paragraph_dict["clean_content"].strip()
    text_tokens = [
        (token.text, token.pos_) for token in nlp(text)
    ]
    
    # Get the tags.
    tags = ["O"] * len(text_tokens)
    entities = paragraph_dict["entities"]
    for entity in entities:
        token_offset = entity["token_offset"]
        tokens_length = entity["tokens_length"]
        historical = entity["historical"]
        suffix = "hist" if historical else "not-hist"
        for i in range(tokens_length):
            if i == 0:
                tags[token_offset + i] = "B-" + suffix
            else:
                tags[token_offset + i] = "I-" + suffix

    # Split the paragraph in sentences shorter than 
    # a given number of tokens in order to match the  
    # pretrained BERT max input length. 
    offsets = get_sentences_offsets(text_tokens, par_title, par_num)

    tokens_results = []
    tags_results = []
    for min_off, max_off in offsets:
        tokens_results.append(
            [token for token, pos in text_tokens[min_off:max_off]]
        )
        tags_results.append(tags[min_off:max_off])

    return tokens_results, tags_results


def get_sentences_offsets(
    text_tokens: List[Tuple],
    par_title: str, 
    par_num: int,
    threshold: int = 256
):
    """BERT allows a maximum of 512 tokens in input, so we need
    to split paragraphs that are longer than 512 tokens into
    smaller paragraphs. We use periods (".") to get the
    boundaries of the sentences and we take the largest 
    sequences that contain a number of tokens smaller than a
    fixed threshold. This threshold is lower than 512, 
    because the BERT tokenizer will split some words into
    subwords, thus incrementing the total number of tokens.
    """

    offset = 0
    max_offset = min(threshold, len(text_tokens))
    length_left = len(text_tokens)

    offsets = []

    while length_left > 0:
        for i in reversed(range(max_offset)):
            token, pos = text_tokens[i]

            # We save the sequence when we find a period, or the sequence is as long
            # as the remainder of the paragraph, or we could not find a period in the
            # whole section of the paragraph.
            if ((token == ".") and (pos == "PUNCT")) or (i == (len(text_tokens) - 1)) \
                or (i == offset):
                new_offset = i + 1
                offsets.append((offset, new_offset))
                max_offset = min(new_offset + threshold, len(text_tokens))
                offset = new_offset

                length_left = len(text_tokens) - offset

                break

    return offsets


def get_text_tokens_tags(text, tags, tokenizer):
    text_tokens_tags = []

    for word, tag in zip(text, tags):

        tokenized_word = tokenizer.tokenize(word)
        text_tokens_tags.extend([tag] * len(tokenized_word))

    return text_tokens_tags


@Cache(
    cache_path=[
        "cache/h_events/{function_name}/{limit}/texts_{_hash}.pkl",
        "cache/h_events/{function_name}/{limit}/tags_{_hash}.pkl",
        "cache/h_events/{function_name}/{limit}/labels_{_hash}.pkl"
    ],
    args_to_ignore=["dataset", "nlp"]
)
def load_data(
    dataset: dict,
    nlp: spacy.language.Language,
    limit: int = None
):
    texts = []
    tags = []
    labels = []

    if limit == None:
        limit = len(dataset["paragraphs"])

    # Here we use the entities in each paragraph to
    # create the tags for the paragraphs.
    for paragraph in tqdm(
        dataset["paragraphs"][:limit],
        desc="Creating tags",
        leave=False
    ):
        par_num = paragraph["par_num"]
        par_title = paragraph["title"]
        text_tokens, text_tokens_tags = tag_paragraph(
            paragraph, nlp, par_title, par_num
        )
        texts.extend(text_tokens)
        tags.extend(text_tokens_tags)
        for _ in range(len(text_tokens)):
            labels.append(paragraph["historical"])

    texts = np.array(texts, dtype=object)
    tags = np.array(tags, dtype=object)
    labels = np.array(labels, dtype=object)

    assert len(texts) == len(tags)
    assert len(labels) == len(tags)

    return texts, tags, labels


def preprocess_data(
    texts: List[str],
    tags: List[str],
    labels: List[int],
    tokenizer: transformers.BertTokenizerFast,
    ignore_other: bool = True,
    padding: Union[str, bool] = True
):
    # We create utility dictionaries tag2dix and idx2tag to convert
    # tag index to label and vice versa.
    unique_tags = sorted(set(tag for text_tags in tags for tag in text_tags))
    tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    tag2idx["MASK"] = -100
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    if ignore_other:
        del idx2tag[tag2idx["O"]]
        tag2idx["O"] = -100

    # Now we need to use the BERT tokenizer to get the real tokens,
    # not the Spacy tokens. BERT may split a word in multiple subwords,
    # so we need to adjust the tags. We repeat the tag of the word for
    # each of the subwords. 
    # Note that the BERT tokenizer here adds the special tokens [CLS]
    # and [SEP] and will also pad the text up to 512 tokens.
    # Set "padding" to "max_length" to always pad to 512 tokens.
    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        padding=padding, 
        truncation=True
    )

    tokens_labels = [] 

    for i, (text, text_tags) in tqdm(
        enumerate(zip(texts, tags)),
        desc="Adjusting tags to encodings",
        leave=False
    ):
        text_tokens_tags = get_text_tokens_tags(text, text_tags, tokenizer)
        text_enc_tokens_labels = np.ones(
            len(encodings[i]), dtype=int
        ) * -100

        try:
            text_enc_tokens_labels[1:len(text_tokens_tags) + 1] = [
                tag2idx[tag] for tag in text_tokens_tags
            ]
        except Exception as ex:
            traceback.print_exc()
            print("encodings", i)
            print("text", text)
            print("text tokens tags", text_tokens_tags)
            print(text_enc_tokens_labels.shape, torch.tensor(text_tokens_tags).shape)
            raise ex
        tokens_labels.append(text_enc_tokens_labels.tolist())

    if ignore_other:
        # We have already converted all "O" to "MASK",
        # so we can remove the corresponding keys from 
        # the dictionary.
        del tag2idx["O"]

    return encodings, tokens_labels, labels, tag2idx, idx2tag


def tensor_accuracy(y_true, y_preds):

    # The model outputs probabilities for each token in the
    # sentence. So the output for the tokens will be of size
    # (#batches, #tokens, #classes). Method torch.max returns
    # the max value and the index of the max value. So we can get
    # the class predicted for each token in the batch.

    # _, y_preds = torch.max(y_preds_probas, -1)
    y_correct = (y_preds == y_true).sum().detach()
    acc = y_correct / y_true.size(0)

    return acc


def tensor_binary_accuracy(y_true, y_preds_probas):
    y_preds = (y_preds_probas > 0.5).view(1, -1)
    y_correct = (y_preds == y_true).sum().detach()
    acc = y_correct / y_true.size(0)

    return acc


def get_spans(
    tokens: List[str], 
    last_token_idx: int,
    span_max_length: int = 8
):
    # We use last_token_idx in case the original sentence
    # was truncated by the BERT tokenizer. In this case we
    # need to stop creating spans at an earlier index than
    # the length of the original sentence.
    if last_token_idx is None:
        last_token_idx = len(tokens)

    last_index = min(len(tokens), last_token_idx)

    spans = []
    for i in range(last_index):
        for j in range(i, min(last_index, i + span_max_length)):
            spans.append((i, j, j - i + 1))
    return spans


def get_mapper(
    offset_mapping, 
    input_ids, 
    tokens: List[str],
    tokenizer,
    doc_evts,
    doc_args
):
    """Return a dictionary that maps each token in the 
    original sentence to the corresponding indices in the
    sentence tokenized by the BERT tokenizer, which might
    split a word into sub words.

    In the dictionary, for each token, we obtain a tuple 
    corresponding to the start and end indices of the token
    in the tokenized sentence.

    We also return a likewise dictionary for the reverse
    operation.
    """
    token2idx = {}
    idx2token = {}
    idx = -1

    for off_num, offsets in enumerate(offset_mapping):
        # Special tokens, like CLS, SEP or PAD. We skip
        # them, since they aren't in the original
        # sentence.
        if offsets[1] == 0:
            continue

        # The token was split in sub words. The start
        # index for the original token will be the same,
        # but we need to update the end index.
        if offsets[0] != 0:
            token2idx[idx][1] = off_num
            
        # Normal tokens.
        else:
            idx = idx + 1
            token2idx[idx] = [off_num, off_num]

    start2token = {start: tok for tok, [start, end] in token2idx.items()}
    end2token = {end: tok for tok, [start, end] in token2idx.items()}
    idx2token = {
        "start": start2token,
        "end": end2token
    }

    last_token_idx = None

    # Sanity check.
    for i, token in enumerate(tokens):
        if i >= len(token2idx):
            print(
                "The original sentence was", len(tokens),
                "tokens long and was tokenized by BERT into",
                f"{len(input_ids)} tokens.", "The sentence was",
                f"truncated at index {i}. The token2idx dict",
                f"had length {len(token2idx)}." 
            )
            
            last_token_idx = i - 1

            # Check that all events and arguments are within
            # the truncated sentence.
            for evt in doc_evts:
                evt_start = evt[0]
                evt_end = evt[1]
                if evt_start >= i or evt_end >= i:
                    raise Exception(
                        f"Event {evt} was in a truncated"
                        "part of a sentence."
                    )

            for arg in doc_args:
                arg_start = arg[0]
                arg_end = arg[1]
                if arg_start >= i or arg_end >= i:
                    raise Exception(
                        f"Argument {arg} was in a truncated"
                        "part of a sentence."
                    )

            print("doc_evts", doc_evts)
            print("doc_args", doc_args)
            break
        try:
            ids = input_ids[
                token2idx[i][0]:
                token2idx[i][1] + 1
            ]
        except Exception as ex:
            traceback.print_exc()
            print("token number", i)
            print("input ids", input_ids)
            print("token2idx", token2idx)
            print("tokens", tokens)
            print("tokens len", len(tokens))
            print("input ids len", len(input_ids))
            print("token2idx len", len(token2idx))
            print("decoded ids", tokenizer.decode(input_ids))
            print("offset_mapping", offset_mapping)
            print("len offset_mapping", len(offset_mapping))

        decoded_token = tokenizer.decode(ids)

        if (token != decoded_token) and (decoded_token in token):
            # It is possible that the last token is the one being
            # splitted in sub words and some sub words may have
            # exceeded the 512 token limit. 
            print("Last token was splitted: ", token, decoded_token)
            continue
        try:
            assert token == decoded_token.replace(" ", ""), \
                (token, decoded_token)
        except AssertionError as ae:
            print("token number", i)
            print("input ids", input_ids)
            print("token2idx", token2idx)
            print("tokens", tokens)
            print("tokens len", len(tokens))
            print("input ids len", len(input_ids))
            print("token2idx len", len(token2idx))
            print("decoded ids", tokenizer.decode(input_ids))
            print("offset_mapping", offset_mapping)
            print("len offset_mapping", len(offset_mapping))
            raise ae

    return token2idx, idx2token, last_token_idx


def load_rams_data(
    base_path: str = "datasets/rams",
    split: str = "train",
):
    docs = []
    events = []
    arguments = []
    evt2idx = {}
    idx2evt = {}
    arg2idx = {}
    idx2arg = {}

    with open(
        f"{base_path}/raw/{split}.jsonlines",
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            docs.append(obj)
            events.append(obj["evt_triggers"][0][2][0][0])
            for ent in obj["ent_spans"]:
                arguments.append(ent[2][0][0])

    events = sorted(set(events))
    arguments = sorted (set(arguments))

    evt2idx = {evt: idx for idx, evt in enumerate(["no_evt"] + events)}
    idx2evt = {idx: evt for evt, idx in evt2idx.items()}
    arg2idx = {arg: idx for idx, arg in enumerate(["no_arg"] + arguments)}
    idx2arg = {idx: arg for arg, idx in arg2idx.items()}

    dicts = {
        "evt2idx": evt2idx,
        "idx2evt": idx2evt,
        "arg2idx": arg2idx,
        "idx2arg": idx2arg
    }

    return docs, dicts


def get_rams_data_dict(
    docs: List[dict],
    tokenizer,
    split: str,
    span_max_length: int = 3,
    base_path: str = "datasets/rams/preprocessed",
    write: bool = True,
    map_dicts: dict = None
):
    evt2idx = map_dicts["evt2idx"]
    arg2idx = map_dicts["arg2idx"]

    spans = []
    spans_trg_labels = []
    spans_arg_labels = []
    tokens = []
    tokens_ids = []
    attention_masks = []
    triggers = []
    arguments = []
    doc_keys = []
    span_mappers = []

    args_ids = []
    args_masks = []
    args_dec_ids = []
    args_dec_masks = []

    argument_tokenizer = BartTokenizer.from_pretrained(
        "facebook/bart-base"
    )
    argument_tokenizer.add_tokens([" <arg>", " <trg>"])

    for doc in tqdm(
        docs,
        desc="Processing document",
        leave=False
    ):
        doc_keys.append(doc["doc_key"])
        doc_tokens = sum(doc["sentences"], [])
        fixed_doc_tokens = []

        for tok in doc_tokens:
            if "\u200b" in tok or "\u200e" in tok:
                if len(tok) == 1:
                    tok = tok.replace("\u200b", "-")
                    tok = tok.replace("\u200e", "-")
                else:
                    tok = tok.replace("\u200b", "")
                    tok = tok.replace("\u200e", "")
            tok = tok.replace("\u201c", "\"").replace("\u201d", "\"") \
                    .replace("\u2019", "'").replace("\u2014", "⁠—").replace("\u2060", "") \
                    .replace("\xad", "").replace("\x9a", "").replace("\x7f", "-") \
                    .replace("\x93", "\"").replace("\x94", "\"").replace("\x96", "–") \
                    .replace("\x92", "'").replace("☰", "-").replace("📸", "-") \
                    .replace("▪", "-").replace("👎🏻", "-").replace("�", "-") \
                    .replace("\ufffd", "-").replace("\x9d", "")
            fixed_doc_tokens.append(tok)

        doc_tokens = fixed_doc_tokens
        tokens.append(doc_tokens)

        # ========================
        # EVENT/TRIGGET EXTRACTION
        # ========================

        tokenizer_output = tokenizer(
            doc_tokens,
            is_split_into_words=True,
            padding="max_length", 
            truncation=True,
            return_offsets_mapping=True
        )
        tokens_ids.append(
            tokenizer_output.input_ids
        )
        attention_masks.append(
            tokenizer_output.attention_mask
        )

        token2idx, idx2token, last_token_idx = get_mapper(
            tokenizer_output.offset_mapping,
            tokenizer_output.input_ids,
            doc_tokens,
            tokenizer,
            doc["evt_triggers"],
            doc["ent_spans"]
        )

        # Since we are using doc_tokens and not the tokenizer
        # output, we will take spans only up to the original
        # sentence length. We will not take spans over the
        # padding tokens.
        doc_spans = get_spans(
            doc_tokens, 
            last_token_idx,
            span_max_length=span_max_length
        )

        # doc_spans contains the start and end index
        # of the span in the original tokens.
        span_mappers.append(doc_spans)

        doc_spans = [
            [token2idx[span[0]][0], token2idx[span[1]][1]]
            for span in doc_spans
        ]
        spans.append(doc_spans)

        doc_trigger = doc["evt_triggers"]
        # We retrieve the index in the original sentence, we
        # transform this into the index resulting from 
        # the BERT tokenizer and we pick the first index,
        # which is the index of the first sub word if the
        # token was split. For trigger_end we pick the 
        # second index, which is the index of the last sub
        # word.
        trigger_start = token2idx[doc_trigger[0][0]][0]
        trigger_end = token2idx[doc_trigger[0][1]][1]
        trigger_name = doc_trigger[0][2][0][0]
        trigger_label = evt2idx[trigger_name]
            
        triggers.append(
            [trigger_start, trigger_end, trigger_label]
        )

        # We assign a label to each span. If it is a trigger
        # we assign the label of the corresponding trigger,
        # otherwise we assign the "no_trg" label.
        doc_spans_trg_labels = []
        for span in doc_spans:
            # print("span", span)
            # print("trigger_start", trigger_start)
            # print("trigger_end", trigger_end)
            # print("trigger_label", trigger_label)
            if (span[0] == trigger_start) and \
                (span[1] == trigger_end):
                doc_spans_trg_labels.append(
                    trigger_label
                )
            else:
                doc_spans_trg_labels.append(
                    evt2idx["no_evt"]
                )

        # We should have at least one event.
        assert sum(doc_spans_trg_labels) > 0
        # We should have a label for each span.
        assert len(doc_spans) == len(doc_spans_trg_labels)
        spans_trg_labels.append(doc_spans_trg_labels)

        doc_arguments = []
        for argument in doc["ent_spans"]:
            arg_start = token2idx[argument[0]]
            arg_end = token2idx[argument[1]]
            arg_label = arg2idx[argument[2][0][0]]
            doc_arguments.append(
                [arg_start, arg_end, arg_label]
            )
        arguments.append(doc_arguments)

        # We assign a label to each span for the
        # arguments, like we did for the triggers.
        doc_spans_arg_labels = []
        for span in doc_spans:
            if (span[0] == trigger_start) and \
                (span[1] == trigger_end):
                doc_spans_arg_labels.append(
                    trigger_label
                )
            else:
                doc_spans_arg_labels.append(
                    evt2idx["no_evt"]
                )
        assert len(doc_spans) == len(doc_spans_arg_labels), \
            (len(doc_spans), len(doc_spans_arg_labels))
        spans_arg_labels.append(doc_spans_arg_labels)

        # ===================
        # ARGUMENT EXTRACTION
        # ===================

        doc_trigger = doc["evt_triggers"]
        trigger_start = doc_trigger[0][0]
        trigger_end =doc_trigger[0][1]
        trigger_name = doc_trigger[0][2][0][0]

        ontology_dict = load_ontology()
        template = ontology_dict[
            trigger_name.replace("n/a", "unspecified")
        ]["template"]

        template_in = template2tokens(
            template, argument_tokenizer
        )

        for argument in doc["ent_spans"]:
            arg_start = argument[0]
            arg_end = argument[1]
            arg_name = argument[2][0][0]
            arg_num = ontology_dict[
                trigger_name.replace("n/a", "unspecified")
            ][arg_name]
            arg_text = " ".join(
                doc_tokens[arg_start : arg_end + 1]
            )
            template = re.sub(f"<{arg_num}>", arg_text, template)

        template_out = template2tokens(
            template, argument_tokenizer
        )
        
        prefix = argument_tokenizer.tokenize(
            " ".join(doc_tokens[:trigger_start]),
            add_prefix_space=True
        )
        trg = argument_tokenizer.tokenize(
            " ".join(doc_tokens[trigger_start:trigger_end+1]),
            add_prefix_space=True
        )
        suffix = argument_tokenizer.tokenize(
            " ".join(doc_tokens[trigger_end+1:]),
            add_prefix_space=True
        )
        context = prefix + [" <trg>", ] + trg + [" <trg>", ] + suffix
        
        arg_in = argument_tokenizer.encode_plus(
            template_in, 
            context, 
            add_special_tokens=True,
            add_prefix_space=True,
            max_length=424,
            truncation="only_second",
            padding="max_length"
        )

        arg_out = argument_tokenizer.encode_plus(
            template_out, 
            add_special_tokens=True,
            add_prefix_space=True, 
            max_length=72,
            truncation=True,
            padding="max_length"
        )
        
        args_ids.append(arg_in["input_ids"])
        args_masks.append(arg_in["attention_mask"])
        args_dec_ids.append(arg_out["input_ids"])
        args_dec_masks.append(arg_out["attention_mask"])

    result = {
        "spans": spans,
        "spans_trg_labels": spans_trg_labels, 
        "spans_arg_labels": spans_arg_labels, 
        "tokens": tokens,
        "tokens_ids": tokens_ids,
        "attention_masks": attention_masks, 
        "triggers": triggers,
        "arguments": arguments, 
        "map_dicts": map_dicts,
        "args_ids": args_ids,
        "args_masks": args_masks,
        "args_dec_ids": args_dec_ids,
        "args_dec_masks": args_dec_masks,
        "doc_keys": doc_keys,
        "span_mappers": span_mappers
    }

    if write:
        os.makedirs(base_path, exist_ok=True)

        with open(
            f"{base_path}/{split}.json", "w"
        ) as f_out:
            json.dump(result, f_out)

    return result


def load_ontology(
    base_path: str = "datasets"
):
    ontology = pd.read_csv(f"{base_path}/aida_ontology_cleaned.csv")
    ontology_dict = dict()

    for event_type in ontology["event_type"]:
        ontology_dict[event_type] = dict()
        
        row = ontology[ontology["event_type"] == event_type]
        ontology_dict[event_type]["template"] = row[
            "template"
        ].values[0]
        for arg_num, arg in zip(
            row.iloc[0,2:].index, 
            row.iloc[0, 2:]
        ):
            if isinstance(arg, str):
                ontology_dict[event_type][arg] = arg_num
                ontology_dict[event_type][arg_num] = arg

    return ontology_dict


def template2tokens(
    template: str,
    tokenizer: transformers.BartTokenizerFast
):
    template = re.sub(
        r"<arg\d>", "<arg>", template
    ).split(" ")

    template_tokens = []
    for word in template:
        template_tokens.extend(
            tokenizer.tokenize(
                word,
                add_prefix_space=True 
            )
        )

    return template_tokens


def sanity_check_preprocessed_data(
    split: str = "train",
    dm = None
):
    bart_tokenizer = BartTokenizer.from_pretrained(
        "facebook/bart-base"
    )
    bart_tokenizer.add_tokens([" <arg>", " <trg>"])

    bert_tokenizer = BertTokenizer.from_pretrained(
        "bert-base-cased"
    )

    if split == "train":
        dl = dm.train_dataloader()
    elif split == "valid":
        dl = dm.val_dataloader()
    elif split == "test":
        dl = dm.test_dataloader()
    else:
        raise ValueError

    docs =[]

    if split == "valid":
        split = "dev"

    with open(
            f"datasets/rams/raw/{split}.jsonlines",
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            docs.append(obj)

    it = iter(dl)
    for item in tqdm(
        it,
        total=len(docs),
        desc="Checking document",
        leave=False
    ):
        doc_key = item["doc_keys"][0]
        for d in docs:
            if d["doc_key"] == doc_key:
                doc = d
                break

        spans_trg_true = item["spans_trg_true"]
        spans = item["spans"][0]
        input_ids = item["input_ids"][0]
        idx = torch.argmax(spans_trg_true)
        span_start, span_end = spans[idx]

        evt_text_preproc = bert_tokenizer.decode(
            input_ids[span_start:span_end +1]
        )

        doc_sentences = doc["sentences"]
        sent = sum(doc_sentences, [])

        evt = doc["evt_triggers"][0]
        evt_start, evt_end = evt[0], evt[1]
        evt_text_doc = " ".join(sent[evt_start:evt_end + 1])

        assert evt_text_preproc == evt_text_doc, (
            evt_text_preproc, evt_text_doc
        )

        evt_span_idx = torch.argmax(spans_trg_true)
        evt_name = evt[2][0][0]

        enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_train(
            input_ids=item["input_ids"],
            span_batch=item["spans"],
            spans_with_evts=[evt_span_idx],
            evt_names=[evt_name],
            doc_tokens=item["doc_tokens"],
            span_mappers=item["span_mappers"],
            bart_tokenizer=bart_tokenizer
        )

        enc_text = bart_tokenizer.decode(
            enc_ids[0]
        )

        item_enc_text = bart_tokenizer.decode(
            item["encoder_input_ids"][0]
        )

        assert enc_text == item_enc_text, \
            (enc_text, item_enc_text)
        assert torch.tensor(enc_ids)[0].tolist() == item["encoder_input_ids"][0].tolist(), \
            (torch.tensor(enc_ids), item["encoder_input_ids"])
        assert torch.tensor(enc_attn_masks)[0].tolist() == item["encoder_attention_mask"][0].tolist(), \
            (torch.tensor(enc_attn_masks), item["encoder_attention_mask"])


def remove_special_tokens(text: str):
    text = text.replace("[CLS]", "")
    text = text.replace("[SEP]", "")
    text = text.replace("[PAD]", "")
    # text = text.replace("- -", "--")
    return text.strip()


def get_bart_sentences_train(
    input_ids, span_batch, 
    spans_with_evts, evt_names,
    doc_tokens, span_mappers,
    bart_tokenizer,
    ontology_base_path: str = "datasets"
):
    enc_ids = []
    enc_attn_masks = []
    enc_sentences = []

    for idx, (span_list, evt_span_idx, batch_ids, evt_name) in enumerate(
        zip(
            span_batch, spans_with_evts, input_ids, evt_names
        )
    ):
        tokens = doc_tokens[idx]
        span_mapper = span_mappers[idx]
        assert evt_span_idx < len(span_list), (evt_span_idx, len(span_list))
        original_evt_span = span_mapper[evt_span_idx]
        original_span_start, original_span_end, _ = original_evt_span

        ontology_dict = load_ontology(ontology_base_path)
        template = ontology_dict[
            evt_name.replace("n/a", "unspecified")
        ]["template"]

        template_in = template2tokens(
            template, bart_tokenizer
        )

        prefix = bart_tokenizer.tokenize(
            " ".join(tokens[:original_span_start]),
            add_prefix_space=True
        )

        trg = bart_tokenizer.tokenize(
            " ".join(
                tokens[
                    original_span_start:original_span_end+1
                ]
            ),
            add_prefix_space=True
        )

        suffix = bart_tokenizer.tokenize(
            " ".join(tokens[original_span_end+1:]),
            add_prefix_space=True
        )

        context = prefix + [" <trg>", ] + trg + [" <trg>", ] + suffix

        arg_in = bart_tokenizer.encode_plus(
            template_in, 
            context, 
            add_special_tokens=True,
            add_prefix_space=True,
            max_length=424,
            truncation="only_second",
            padding="max_length"
        )

        enc_id = arg_in["input_ids"]
        enc_sentence = bart_tokenizer.decode(
            enc_id
        )

        enc_ids.append(
            enc_id
        )
        enc_attn_masks.append(
            arg_in["attention_mask"]
        )
        enc_sentences.append(
            enc_sentence
        )

    return enc_ids, enc_attn_masks, enc_sentences


def get_bart_sentences_not_train(
    input_ids, span_batch, 
    spans_with_evts, evt_names,
    bert_tokenizer, bart_tokenizer,
    ontology_base_path: str = "datasets"
):
    enc_ids = []
    enc_attn_masks = []
    enc_sentences = []
    ontology_dict = load_ontology(ontology_base_path)

    for span_list, evt_span_idx, batch_ids, evt_name in zip(
        span_batch, spans_with_evts, input_ids, evt_names
    ):
        template = ontology_dict[
            evt_name.replace("n/a", "unspecified")
        ]["template"]
        template_in = template2tokens(
            template, bart_tokenizer
        )

        evt_span = span_list[evt_span_idx]
        span_start, span_end = evt_span

        prefix_text = bert_tokenizer.decode(
            batch_ids[:span_start]
        )
        prefix_text = remove_special_tokens(prefix_text).strip()
        prefix = bart_tokenizer.tokenize(
            prefix_text,
            add_prefix_space=True
        )

        trg_text = bert_tokenizer.decode(
            batch_ids[span_start : span_end + 1]
        )
        trg_text = remove_special_tokens(trg_text).strip()
        trg = bart_tokenizer.tokenize(
            trg_text,
            add_prefix_space=True
        )

        suffix_text = bert_tokenizer.decode(
            batch_ids[span_end + 1:]
        )
        suffix_text = remove_special_tokens(suffix_text).strip()
        suffix = bart_tokenizer.tokenize(
            suffix_text,
            add_prefix_space=True
        )

        context = prefix + [" <trg>", ] + trg + [" <trg>", ] + suffix

        arg_in = bart_tokenizer.encode_plus(
            template_in, 
            context, 
            add_special_tokens=True,
            add_prefix_space=True,
            max_length=424,
            truncation="only_second",
            padding="max_length"
        )

        enc_id = arg_in["input_ids"]
        enc_sentence = bart_tokenizer.decode(
            enc_id
        )

        enc_ids.append(
            enc_id
        )
        enc_attn_masks.append(
            arg_in["attention_mask"]
        )
        enc_sentences.append(
            enc_sentence
        )

    return enc_ids, enc_attn_masks, enc_sentences


def get_bart_sentences_from_preprocessed(
    input_ids, span_batch, 
    spans_with_evts, evt_names,
    bert_tokenizer, bart_tokenizer
):
    enc_ids = []
    enc_attn_masks = []
    enc_sentences = []

    for span_list, evt_span_idx, batch_ids, evt_name in zip(
        span_batch, spans_with_evts, input_ids, evt_names
    ):
        evt_span = span_list[evt_span_idx]
        span_start, span_end = evt_span

        ontology_dict = load_ontology()
        template = ontology_dict[
            evt_name.replace("n/a", "unspecified")
        ]["template"]

        template_in = template2tokens(
            template, bart_tokenizer
        )

        prefix_text = bert_tokenizer.decode(
            batch_ids[:span_start]
        )
        prefix_text = remove_special_tokens(prefix_text)
        prefix = bart_tokenizer.tokenize(
            prefix_text,
            add_prefix_space=True
        )

        trg_text = bert_tokenizer.decode(
            batch_ids[span_start : span_end + 1]
        )
        trg_text = remove_special_tokens(trg_text)
        trg = bart_tokenizer.tokenize(
            trg_text,
            add_prefix_space=True
        )

        suffix_text = bert_tokenizer.decode(
            batch_ids[span_end + 1:]
        )
        suffix_text = remove_special_tokens(suffix_text)
        suffix = bart_tokenizer.tokenize(
            suffix_text,
            add_prefix_space=True
        )

        context = prefix + [" <trg>", ] + trg + [" <trg>", ] + suffix

        arg_in = bart_tokenizer.encode_plus(
            template_in, 
            context, 
            add_special_tokens=True,
            add_prefix_space=True,
            max_length=424,
            truncation="only_second",
            padding="max_length"
        )

        enc_id = arg_in["input_ids"]
        enc_sentence = bart_tokenizer.decode(
            enc_id
        )

        enc_ids.append(
            enc_id
        )
        enc_attn_masks.append(
            arg_in["attention_mask"]
        )
        enc_sentences.append(
            enc_sentence
        )

    return enc_ids, enc_attn_masks, enc_sentences


def remove_det_prefix_ls(
    text_list: List[str],
    full_text: List[str],
    pointer: int
):  
    try:
        if len(text_list) == 0:
            print("Empty arg")
            print(text_list)
            print(full_text)
            print(pointer)
            return text_list
        else:
            prefixes = ["the", "an", "a"]
            if text_list[0].lower() in prefixes:
                return text_list[1:]
    except Exception as ex:
        print(text_list)
        print(full_text)
        print("pointer", pointer)
        raise ex

def remove_det_prefix_str(
    text: str
):
    prefixes = ["the ", "The ", "an ", "An ", "a ", "A "]
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text 


def find_matches(pred, template, evt_type, ontology_dict):
    """Code loosely based on:
    https://github.com/raspberryice/gen-arg/blob/1f547018f078aeb6fbcdf7a7a11366a77a53fc7e/src/genie/scorer.py
    """

    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    template_words = _RE_COMBINE_WHITESPACE.sub(" ", template).strip().split()
    predicted_words = _RE_COMBINE_WHITESPACE.sub(" ", pred["predicted"]).strip().split()
    gold_words = _RE_COMBINE_WHITESPACE.sub(" ", pred["gold"]).strip().replace("\s", " ").split()  
    predicted_args = defaultdict(list) # each argname may have multiple participants 
    gold_args = defaultdict(list)
    t_ptr= 0
    p_ptr= 0
    g_ptr = 0
    correct_num = 0
    missing_num = 0
    overpred_num = 0
    while t_ptr < len(template_words) \
        and p_ptr < len(predicted_words) \
        and g_ptr < len(gold_words):
        if re.match(r"<(arg\d+)>", template_words[t_ptr]):
            m = re.match(r"<(arg\d+)>", template_words[t_ptr])
            arg_num = m.group(1)
            try:
                arg_name = ontology_dict[evt_type][arg_num]
            except KeyError:
                print(evt_type)
                exit() 

            if predicted_words[p_ptr] == "<arg>":
                # No prediction for this argument.
                if gold_words[g_ptr] == "<arg>":
                    # No gold argument either.
                    gold_args[arg_name].append(None)
                    g_ptr+=1
                else:
                    # Gold template had the argument, but
                    # the prediction missed it.
                    missing_num += 1

                    gold_arg_start = g_ptr
                    while (g_ptr < len(gold_words)) \
                        and (
                            (t_ptr == len(template_words)-1)
                            or
                            (gold_words[g_ptr] != template_words[t_ptr+1])
                        ):
                        g_ptr+=1 
                    gold_arg_text = gold_words[gold_arg_start:g_ptr]
                    gold_arg_text = remove_det_prefix_str(" ".join(gold_arg_text))
                    gold_args[arg_name].append(gold_arg_text)

                predicted_args[arg_name].append(None)
                p_ptr += 1 
                t_ptr += 1  
            else:
                # Prediction found an argument.
                pred_arg_start = p_ptr 
                while (p_ptr < len(predicted_words)) \
                    and (
                        (t_ptr == len(template_words)-1)
                        or
                        (predicted_words[p_ptr] != template_words[t_ptr+1])
                    ):
                    p_ptr += 1 
                pred_arg_text = predicted_words[pred_arg_start:p_ptr]
                pred_arg_text = remove_det_prefix_str(" ".join(pred_arg_text))
                predicted_args[arg_name].append(pred_arg_text)

                if gold_words[g_ptr] == "<arg>":
                    # The model overpredicted the argument.
                    overpred_num += 1
                    gold_args[arg_name].append(None)
                    g_ptr+=1

                else:
                    gold_arg_start = g_ptr
                    while (g_ptr < len(gold_words)) \
                        and (
                            (t_ptr == len(template_words)-1)
                            or
                            (gold_words[g_ptr] != template_words[t_ptr+1])
                        ):
                        g_ptr+=1 
                    gold_arg_text = gold_words[gold_arg_start:g_ptr]
                    gold_arg_text = remove_det_prefix_str(" ".join(gold_arg_text))
                    gold_args[arg_name].append(gold_arg_text)

                    if gold_arg_text == pred_arg_text:
                        correct_num += 1
                    else:
                        overpred_num += 1

                t_ptr += 1
                # aligned 
        else:
            t_ptr += 1 
            p_ptr += 1
            g_ptr += 1 

    res = {
        "correct_num": correct_num,
        "missing_num": missing_num,
        "overpred_num": overpred_num,
        "predicted_args": predicted_args, 
        "gold_args": gold_args,
        "template": " ".join(template_words),
        "predicted": " ".join(predicted_words),
        "gold": " ".join(gold_words)
    }

    return res


def evaluate_arguments_results(
    res_dir: str = "results/historical_events/arguments",
    test_filename: str = "datasets/rams/raw/test.jsonlines"
):
    """Precision, recall and F1-score computation is based on
    the official RAMS 1.0 scorer.
    """
    total_correct = 0
    total_missing = 0
    total_overpred = 0

    results = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "details": []
    }

    predictions = []
    test_data = []

    with open(
        f"{res_dir}/predictions.jsonl",
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            predictions.append(obj)

    with open(
        test_filename,
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            test_data.append(obj)
    
    ontology_dict = load_ontology()

    for prediction in predictions:
        doc_key = prediction["doc_key"]
        doc = None
        for d in test_data:
            # print(d["doc_key"], doc_key[0])
            if d["doc_key"] == doc_key[0]:
                doc = d
        if doc is None:
            raise Exception("Document not found.")

        doc_trigger = doc["evt_triggers"]
        event_type = doc_trigger[0][2][0][0]
        event_type = event_type.replace("n/a", "unspecified")
        template = ontology_dict[event_type]["template"]

        match_dict = find_matches(
            prediction, template, event_type, ontology_dict
        )

        total_correct += match_dict["correct_num"]
        total_missing += match_dict["missing_num"]
        total_overpred += match_dict["overpred_num"]

        d_res = {
            "template": match_dict["template"],
            "prediction": match_dict["predicted"],
            "gold": match_dict["gold"],
            "correct": match_dict["correct_num"],
            "missing": match_dict["missing_num"],
            "overpred": match_dict["overpred_num"],
            "predicted_args": match_dict["predicted_args"],
            "gold_args": match_dict["gold_args"] 
        }

        results["details"].append(d_res)

    p = float(total_correct) / float(total_correct + total_missing) if (total_correct + total_missing) else 0.0
    r = float(total_correct) / float(total_correct + total_overpred) if (total_correct + total_overpred) else 0.0
    f1 = 2.0 * p * r  / (p + r) if (p + r) else 0.0

    results["precision"] = p
    results["recall"] = r
    results["f1"] = f1

    with open(
        f"{res_dir}/argument_results.json", 
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)


