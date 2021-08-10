import numpy as np
import torch
import spacy
import traceback
import transformers
from cache_decorator import Cache
from tqdm.notebook import tqdm
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




    

