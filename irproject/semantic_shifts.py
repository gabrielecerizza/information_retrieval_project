import gensim
import hdbscan
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import spacy
import torch
import umap
from cache_decorator import Cache
from collections import defaultdict
from irproject.autoencoder import AutoEncoder
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import Iterable, List

np.random.seed(42)
spacy.prefer_gpu()
device = "cuda" if torch.cuda.is_available() else "cpu"


def remove_pos_tag(word: str):
    return word[:-3]


def pos_to_tag(pos: str):
    pos_to_tag_dict = {
        "ADJ": "aj",
        "NOUN": "nn",
        "PROPN": "pn",
        "VERB": "vb"
    }
    if pos in pos_to_tag_dict:
        return pos_to_tag_dict[pos]
    else:
        # Unknown POS.
        return "uk" 


def tag_to_pos(tag: str):
    tag_to_pos_dict = {
        "aj": "ADJ",
        "nn": "NOUN",
        "pn": "PROPN",
        "vb": "VERB",
        "uk": tag
    }
    return tag_to_pos_dict[tag] 


def load_semeval_targets(
    base_path: str,
    remove_tags: bool = True
):
    target_file = f"{base_path}/targets.txt"

    with open(target_file, "r", encoding="utf-8") as f_in:
        
        targets = [
            line.strip()[:-3] if remove_tags else line.strip()
            for line in f_in
        ]
        return targets


def load_data(
    base_path: str
):
    data = dict()

    with open(
        f"{base_path}/targets.txt", "r", encoding="utf-8"
    ) as f_in:
        semeval_targets = [
            line.strip() for line in f_in
        ]
        data["semeval_targets_tagged"] = semeval_targets
        data["semeval_targets_clean"] = [
            remove_pos_tag(target) for target in semeval_targets
        ]
    
    for corpus_name in ["corpus_old", "corpus_new"]:
        with open(
            f"{base_path}/{corpus_name}.txt", "r", encoding="utf-8"
        ) as f_in:
            data[corpus_name] = [line.strip() for line in f_in]

    return data


def compute_freqs(
    data: dict, 
    load_cache: bool = True, 
    cache_filename: str = "data/semantic_shifts/freqs_dict.pkl"
):
    if load_cache and os.path.isfile(cache_filename):
        with open(cache_filename, "rb") as f_in:
            return pickle.load(f_in)

    nlp = spacy.load("en_core_web_sm")
    freqs = defaultdict(lambda: 0)
    
    for corpus in tqdm(
        [data["corpus_old"], data["corpus_new"]],
        desc="Computing frequencies for corpus",
        leave=False
    ):
        for sentence in tqdm(
            corpus,
            desc="Computing frequencies for sentence",
            leave=False
        ):
            for token in nlp(sentence):
                token_pos = token.pos_
                if (token_pos in ["ADJ", "NOUN", "PROPN", "VERB"]) \
                    and (not token.is_punct) \
                    and (not token.is_stop) \
                    and (token.is_alpha):
                    freqs[
                        token.lemma_ + "_" + pos_to_tag(token_pos)
                    ] += 1

    freqs_dict = dict(
        sorted(
            freqs.items(), key=lambda item: item[1], reverse=True
        )
    )

    with open(cache_filename, "wb") as f_out:
        pickle.dump(
            freqs_dict, f_out, protocol=pickle.HIGHEST_PROTOCOL
        )

    return freqs_dict


def get_targets(data: dict, freqs_dict: dict, n: int = 5000):
    corpora_targets_tagged = list(freqs_dict.keys())[:n]
    corpora_targets_clean = [
        remove_pos_tag(target) for target in corpora_targets_tagged
    ]

    targets_tagged = list(set.union(
        set(corpora_targets_tagged),
        set(data["semeval_targets_tagged"])
    ))
    targets_clean = list(set.union(
        set(corpora_targets_clean),
        set(data["semeval_targets_clean"])
    ))

    targets_tagged = sorted(targets_tagged)
    targets_clean = sorted(targets_clean)

    return targets_tagged, targets_clean


def get_sentences_with_targets(
    data: dict, 
    targets_tagged: List[str],
    load_cache: bool = True, 
    cache_filename: str = "data/semantic_shifts/"
        + "sentences_with_trg.pkl"
):
    if load_cache and os.path.isfile(cache_filename):
        with open(cache_filename, "rb") as f_in:
            return pickle.load(f_in)

    nlp = spacy.load("en_core_web_sm")
    results = dict()

    for corpus_name in tqdm(
        ["corpus_old", "corpus_new"],
        desc="Reading corpus",
        leave=False
    ):
        results[corpus_name] = []
        corpus = data[corpus_name]

        for sentence in tqdm(
            corpus,
            desc="Reading sentence",
            leave=False
        ):
            for token in nlp(sentence):
                token_pos = token.pos_
                tagged_token = token.lemma_ + "_" \
                    + pos_to_tag(token_pos)
                if tagged_token in targets_tagged:
                    results[corpus_name].append(sentence)
                    break

    with open(cache_filename, "wb") as f_out:
        pickle.dump(
            results, f_out, protocol=pickle.HIGHEST_PROTOCOL
        )

    return results


def remove_from_spacy_list(ls, word):
    result = []
    for (i, token) in enumerate(ls):
        if token.text == word:
            return result + ls[i + 1:]
        else:
            result.append(token)


def get_sublist_indices(
    ls, 
    availability,
    subls
):
    len1, len2 = len(ls), len(subls)
    for i in range(len1):
        if (ls[i:i + len2] == subls) \
            and all(availability[i:i + len2]):
            availability[i:i + len2] = [False] * len2
            return (i, i + len2), availability


def tokenize_sentences(
    sentences: List[str],
    tokenizer: AutoTokenizer, 
    targets_tagged: List[str],
    targets_clean: List[str]
):
    nlp = spacy.load("en_core_web_sm")

    for sentence in sentences:
        
        tokenized_sentence = tokenizer.tokenize(
            sentence, add_special_tokens=True, 
            truncation=True, max_length=512
        )
        tokens_ids = tokenizer.encode(
            sentence, truncation=True, max_length=512
        )

        # Needed to handle duplicate targets.
        tokens_availability = [True] * len(tokens_ids)

        spacy_tokens = [
            sp_token for sp_token in nlp(sentence)
        ]

        targets_indices = {
            target: []
            for target in targets_tagged
        }

        for spacy_token in spacy_tokens:
            spacy_token_lemma = spacy_token.lemma_
            if spacy_token.lemma_ in targets_clean:
                target = "".join(
                    [spacy_token_lemma, "_", 
                    pos_to_tag(spacy_token.pos_)]
                )
                if target in targets_tagged:

                    # We need to find the indices of the target
                    # word as it was encoded by the BERT
                    # tokenizer. The BERT tokenizer might split
                    # a word in multiple character n-grams.
                    # For example, "embedding" is tokenized as
                    # ['em', '##bed', '##ding'] and the
                    # corresponding ids are [9712, 4774, 3408].
                    spacy_token_ids = tokenizer.encode(
                        spacy_token.text, add_special_tokens=False
                    )
                    try:
                        target_indices, tokens_availability = get_sublist_indices(
                            tokens_ids,
                            tokens_availability, 
                            spacy_token_ids
                        )
                        targets_indices[target].append(target_indices)
                    except TypeError:
                        # Skipping token artifacts. They are mostly
                        # single characters resulting from different
                        # tokenization between spacy and BERT.
                        pass
                        # print("Skipping possible artifact "
                        #    + f"token {spacy_token}, "
                        #    + f"target {target}, "
                        #    + f"sentence {sentence}")
                        # print("Unexpected error:", sys.exc_info()[0])
                        # print("Sentence: ", sentence)
                        # print("Spacy token: ", spacy_token)
                        # print("Target: ", target)
                        # print("spacy_token_ids: ", spacy_token_ids)
                        # print("tokens_ids", tokens_ids)
                        # print("tokens_availability: ", tokens_availability)

        yield tokenized_sentence, targets_indices


def compute_embeddings(
    model: AutoModel, 
    tokens_ids: torch.tensor, 
    targets_indices: dict
):
    targets_embeddings = {target: [] for target in targets_indices}
    with torch.no_grad():
        outputs = model(
            tokens_ids,
            output_hidden_states=True
        )
        encoded_layers = outputs[2]

        # We use the last layer. Layer 0 is the input.
        embeddings_layer = encoded_layers[12]
        
        for target, target_indices in targets_indices.items():
            # We take the mean when the BERT tokenizer split
            # a word in multiple tokens.
            embedding = [
                torch.mean(
                    embeddings_layer[:, start_index:end_index, :], 
                    dim=1
                ).cpu().numpy().flatten()
                for start_index, end_index in target_indices
            ]
            targets_embeddings[target].extend(embedding)
                
        return targets_embeddings


def compute_targets_embeddings(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    corpus_name: str,
    data: dict,
    tokenized_sentences: Iterable
):
    targets_embeddings_dict = {
        target: [] for target in data["targets_tagged"]
    }
    targets_sentences_dict = {
        target: [] for target in data["targets_tagged"]
    }

    for i, (tokenized_sentence, targets_indices) in tqdm(
        enumerate(tokenized_sentences),
        desc="Computing sentence embeddings",
        leave=False
    ):
        tokens_ids = tokenizer.convert_tokens_to_ids(
            tokenized_sentence
        )
        tokens_ids = torch.tensor([tokens_ids]).to(device)

        targets_embeddings = compute_embeddings(
            model, tokens_ids, targets_indices
        )
        
        for target, embeddings in targets_embeddings.items():
            if len(embeddings) > 0:
                targets_embeddings_dict[target].extend(embeddings)
                targets_sentences_dict[target].extend(
                    [i] * len(embeddings)
                )

    return targets_embeddings_dict, targets_sentences_dict


def save_targets_embeddings(
    targets_embeddings_dict: dict,
    targets_sentences_dict: dict,
    corpus_name: str
):
    base_out_path = f"data/semantic_shifts/embeddings/{corpus_name}"
    os.makedirs(base_out_path, exist_ok=True)

    for target, target_embeddings in tqdm(
        targets_embeddings_dict.items(),
        desc="Saving embeddings and sentences",
        leave=False
    ):
        np.save(f"{base_out_path}/{target}", target_embeddings)
        target_sentences = np.array(targets_sentences_dict[target])
        np.save(f"{base_out_path}/{target}_sen", target_sentences)


def load_target_embeddings(
    target: str,
    corpus_name: str
):
    base_path = f"data/semantic_shifts/embeddings/{corpus_name}"
    return np.load(f"{base_path}/{target}.npy")


def load_target_sentences_indices(
    target: str,
    corpus_name: str
):
    base_path = f"data/semantic_shifts/embeddings/{corpus_name}"
    return np.load(f"{base_path}/{target}_sen.npy")


def train_model( 
        dataloader: DataLoader,
        model: nn.Module, 
        loss_fn: torch.nn.modules.loss._Loss, 
        optimizer: torch.optim.Optimizer
    ):
        for X in dataloader:
            X = X.to(device)

            # Compute prediction error.
            pred = model(X)
            loss = loss_fn(pred, X)

            # Backpropagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@Cache(
    cache_path="cache/{function_name}/{target}/X_encoded_{_hash}.npy",
    args_to_ignore=["X"]
)
def get_autoencoded_embeddings(
    X: np.ndarray,
    target: str
):  
    model = AutoEncoder(
        dims=[X.shape[-1], 500, 500, 2000, 20]
    ).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-5
    )

    train_dataloader = DataLoader(X, batch_size=256)

    epochs = 128
    for _ in range(epochs):
        train_model(
            train_dataloader, model, loss_fn, optimizer
        )

    model.eval()
    X_encoded = []
    for x in X:
        with torch.no_grad():
            pred = model.encoder(torch.from_numpy(x).to(device))
            X_encoded.append(pred.cpu().numpy())

    return X_encoded


@Cache(
    cache_path="cache/{function_name}/{target}_{_hash}.npy",
    args_to_ignore=["X"]
)
def get_umap_embeddings(
    X: np.ndarray,
    target: str
):
    return umap.UMAP(
        n_neighbors=5,
        min_dist=0.0,
        metric="cosine",
        n_components=10
    ).fit_transform(X)

       
def compute_senses_frequencies(
    cluster_labels, 
    embeddings_epochs, 
    embeddings_num_per_epoch
):
    epochs = set(embeddings_epochs)
    cluster_epoch_zip = list(
        zip(cluster_labels, embeddings_epochs)
    )
    senses_frequencies = {epoch: dict() for epoch in epochs}
    for epoch in epochs:
        embeddings_count = embeddings_num_per_epoch[epoch]
        for sense_label in set(cluster_labels):
            sense_count = sum(
                int(
                    cluster_label == sense_label 
                    and epoch == epoch_label
                )
                for cluster_label, epoch_label in cluster_epoch_zip
            )
            sense_frequency = sense_count / embeddings_count
            senses_frequencies[epoch][int(sense_label)] = \
                sense_frequency
    return senses_frequencies


def perform_clustering(
    X: np.ndarray,
    min_cluster_size: float
):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, 
    )
    labels = clusterer.fit_predict(X)
    probas = clusterer.probabilities_

    return labels, probas


def load_vectors(fname, skip_first_line=False):
    fin = io.open(
        fname, 
        "r",
        encoding="utf-8", 
        newline="\n", 
        errors="ignore")
    if skip_first_line:
        fin.readline()
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


def save_word2vec_format(fname, vocab, vector_size, binary=True):
    """Save a file in .txt format into a .bin format that can
    be loaded by Gensim.

    Code adapted from:
    https://stackoverflow.com/questions/45981305/convert-python-dictionary-to-word2vec-object
    """
    
    total_vec = len(vocab)
    with gensim.utils.open(fname, "wb") as fout:
        print(total_vec, vector_size)
        fout.write(
            gensim.utils.to_utf8(
                "%s %s\n" % (total_vec, vector_size)
            )
        )
        # Store in sorted order: most frequent words at the top.
        for word, row in tqdm(vocab.items()):
            if binary:
                row = row.astype(np.float32)
                fout.write(
                    gensim.utils.to_utf8(word) + b" " + row.tostring()
                )
            else:
                fout.write(
                    gensim.utils.to_utf8(
                        "%s %s\n" % \
                        (word, ' '.join(repr(val) for val in row))
                    )
                )

def get_most_freq_targets(
    model_old: gensim.models.KeyedVectors,
    model_new: gensim.models.KeyedVectors,
    semeval_targets: List[str],
    num: int = 5000
):
    nlp = spacy.load("en_core_web_sm")

    targets = []
    model_old_keys = model_old.index_to_key

    for word in set.union(
        set(model_new.index_to_key[:num]),
        set(semeval_targets)
    ):
        # We want the token lemma. It would be better
        # to tokenize the whole sentence to get the
        # correct lemma, but we do not have the corpus,
        # just the embeddings. No way around it.
        token = nlp(word)[0]
        token_lemma = token.lemma_
        if (not token.is_punct) \
            and (token.is_alpha) \
            and (not token.is_stop) \
            and (token.pos_ in ["NOUN", "VERB", "PROPN", "ADJ"]) \
            and (token_lemma in model_old_keys):
            targets.append(token_lemma)
    
    return targets


def procrustes_align_gensim(base_embed, other_embed, words=None):
    """Align other_embed vectors with base_embed vectors
    using Orthogonal Procrustes method.

    Code adapted from:
    https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8 
    """

    in_base_embed, in_other_embed = intersect_vocabulary(
        base_embed, other_embed, words=words
    )

    base_embed.fill_norms(force=True)
    other_embed.fill_norms(force=True)

    base_vecs = in_base_embed.get_normed_vectors()
    other_vecs = in_other_embed.get_normed_vectors()

    m = other_vecs.T.dot(base_vecs) 
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v) 
    other_embed.vectors = (other_embed.vectors).dot(ortho)    
    
    return other_embed


def intersect_vocabulary(
    m1: gensim.models.KeyedVectors,
    m2: gensim.models.KeyedVectors,
    words: List[str] = None,
    normalize: bool = False
):
    """Compute the intersection between the vocabularies of
    two models and replace the vocabulary of both models with
    the computed intersection.

    Code adapted from:
    https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8 
    """
    vocab_m1 = set(m1.index_to_key)
    vocab_m2 = set(m2.index_to_key)

    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    common_vocab = list(common_vocab)
    # We store keys in decreasing order of total frequency 
    # between the two models.
    common_vocab.sort(
        key=lambda w: m1.get_vecattr(w, "count") \
            + m2.get_vecattr(w, "count"), 
        reverse=True
    )

    for m in [m1, m2]:
        indices = [m.key_to_index[w] for w in common_vocab]
        old_arr = m.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.vectors = new_arr

        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.key_to_index = new_key_to_index
        m.index_to_key = new_index_to_key

    if normalize:
        m1.fill_norms(force=True)
        m2.fill_norms(force=True)

        m1.vectors = m1.get_normed_vectors()
        m2.vectors = m2.get_normed_vectors()

    return (m1, m2)


def compute_cosine_shifts(
    model_old: gensim.models.KeyedVectors,
    model_new: gensim.models.KeyedVectors,
    targets: List[str],
    ordered: bool = True
):
    results = dict()

    for target in tqdm(
        targets,
        desc="Computing cosine shifts",
        leave=False
    ):
        results[target] = cosine(
            model_old.get_vector(target), 
            model_new.get_vector(target)
        )

    if ordered:
        return dict(
            sorted(
                results.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        )
    else:
        return results


def nn_shift(
    model_old: gensim.models.KeyedVectors, 
    model_new: gensim.models.KeyedVectors,
    word: str, 
    topn: int = 15
):
    similarities = model_old.similar_by_word(word, topn=topn)
    similar_words, model_old_distances = [], []
    for (similar_word, value) in similarities:
        if similar_word in model_new.index_to_key:
            similar_words.append(similar_word)
            # We want distances, so we subtract the
            # similarity score from 1.
            model_old_distances.append(1 - value)
    
    if len(similar_words) == 0:
        # In model_new the word cannot be found. So the
        # word is only in model_old. We score the shift
        # to be maximum, since the word disappeared
        # during the language evolution. 
        error1 = 1.0
    else:
        model_new_distances = model_new.distances(word, similar_words)
        error1 = mean_squared_error(
            model_old_distances, model_new_distances
        )

    similarities = model_new.similar_by_word(word, topn=topn)
    similar_words, model_new_distances = [], []
    for (similar_word, value) in similarities:
        if similar_word in model_old.index_to_key:
            similar_words.append(similar_word)
            model_new_distances.append(1 - value)

    if len(similar_words) == 0:
        error2 = 1.0
    else:
        model_old_distances = model_old.distances(word, similar_words)
        error2 = mean_squared_error(
            model_old_distances, model_new_distances
        )

    return np.mean([error1, error2])


def compute_nn_shifts(
    model_old: gensim.models.KeyedVectors, 
    model_new: gensim.models.KeyedVectors,
    targets: List[str], 
    topn: int = 15,
    ordered: bool = True
):
    results = dict()

    for target in tqdm(
        targets,
        desc="Computing nearest neighbors shifts",
        leave=False
    ):
        results[target] = nn_shift(
            model_old, model_new, target, topn
        )

    if ordered:
        return dict(
            sorted(
                results.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        )
    else:
        return results


def evaluate_shifts(
    results: dict,
    remove_tags: bool = True
):
    target_file = "datasets/semeval2020/truth/graded.txt"

    y_true, y_pred = [], []

    with open(target_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            spl = line.strip().split("\t")
            if remove_tags:
                token = spl[0][:-3]
            else:
                token = spl[0]
            y_true.append(float(spl[1]))
            y_pred.append(results[token])

    return spearmanr(y_true, y_pred)


def save_static_emb_results(
    results: dict,
    model_old: gensim.models.KeyedVectors,
    model_new: gensim.models.KeyedVectors,
    fname: str,
    topn: int = 100
):
    """Save results from static word embeddings approach."""

    stored_results = dict()

    for target, score in tqdm(
        list(results.items())[:topn],
        desc="Saving results",
        leave=False
    ):
        stored_results[target] = dict()
        stored_results[target]["score"] = score
        stored_results[target]["old_similar_words"] = \
            model_old.similar_by_word(target)
        stored_results[target]["new_similar_words"] = \
            model_new.similar_by_word(target)

    base_out_path = f"results/semantic_shifts"
    os.makedirs(base_out_path, exist_ok=True)

    with open(
        f"{base_out_path}/{fname}.json", "w", 
        encoding="utf-8"
    ) as f_out:
        json.dump(
            stored_results, f_out, 
            ensure_ascii=False, 
            indent=4
        )


def save_context_emb_results(
    results: dict,
    top_sentences: dict,
    one_epoch_targets: List[str],
    hdbscan_errors: List[str],
    memory_issues: List[str],
    targets_senses_frequencies: dict,
    targets_senses_counts: dict,
    fname: str = "jensen_shannon",
    topn: int = 100,
    ordered: bool = True
):
    """Save results from contextualized word embeddings approach."""

    if ordered:
        results = dict(
            sorted(
                results.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        )

    stored_results = {
        "description": "For each target word we provide the "
            + "Jensen Shannon Distance score and a list "
            + "containing (usually) the top 3 sentences for each "
            + "discovered sense of the word. The number of "
            + "sentences for each sense is shown in "
            + "'senses_counts', while the frequencies are " 
            + "shown in 'senses_freqs'. So, the "
            + "first list of three sentences in the list of "
            + "target sentences refers to the sentences of the "
            + "first sense (or sense 0) and so on. Sense -1 is "
            + "noise, so no sentence is provided. The words in "
            + "'one_epoch_target_words' were found only in "
            + "one corpus, either the old or the new one. The "
            + "words in 'hdbscan_errors' could not be processed "
            + "due to errors generated by HDBSCAN. The words in "
            + "'memory_issues' could not be processed due to "
            + "eccessive memory demands. The tags appended to "
            + "the words have the following meaning: 'aj' is "
            + "an adjective; 'nn' is a noun; 'vb' is a verb; "
            + "'pn' is a proper noun. The target words are "
            + "sorted by score.",
        "target_words": dict(),
        "one_epoch_target_words": one_epoch_targets,
        "hdbscan_errors": hdbscan_errors,
        "memory_issues": memory_issues
    }

    for target, score in tqdm(
        list(results.items())[:topn],
        desc="Saving results",
        leave=False
    ):
        stored_results["target_words"][target] = dict()
        stored_results["target_words"][target]["score"] = score
        stored_results["target_words"][target]["senses_counts"] = \
            targets_senses_counts[target]
        stored_results["target_words"][target]["senses_freqs"] = \
            targets_senses_frequencies[target]
        stored_results["target_words"][target]["sentences"] = \
            top_sentences[target]

    base_out_path = f"results/semantic_shifts"
    os.makedirs(base_out_path, exist_ok=True)

    with open(
        f"{base_out_path}/{fname}.json", "w", 
        encoding="utf-8"
    ) as f_out:
        json.dump(
            stored_results, f_out, 
            ensure_ascii=False, 
            indent=4
        )


def plot_targets_senses(
    fname: str = "jensen_shannon",
    topn: int = 100
):
    with open(f"results/semantic_shifts/{fname}.json") as f:
        data = json.load(f)["target_words"]

    sns.set(style = "white")
    base_out_path = "results/semantic_shifts/jensen_shannon_plots"
    eps_path = base_out_path + "/eps"
    png_path = base_out_path + "/png"
    os.makedirs(eps_path, exist_ok=True)
    os.makedirs(png_path, exist_ok=True)

    for idx, target in tqdm(
        enumerate(list(data.keys())[:topn]),
        total=len(list(data.keys())[:topn]),
        desc="Saving plots",
        leave=False
    ):
        target_data = data[target]["senses_counts"]

        df = pd.DataFrame({}, columns=["Corpus", "Sense"])
        for corpus in target_data:
            for sense, count in target_data[corpus].items():
                if int(sense) < 0:
                    continue
                for i in range(count):
                    df = df.append(
                        {"Corpus": corpus, "Sense": sense}, 
                        ignore_index=True
                    )

        # Get the sentences.
        senses_sents = data[target]["sentences"]
        labels = []
        for sentences in senses_sents:
            sentence = sentences[0].strip()
            spl = sentence.split(" ")
            new_spl = ""
            for i, token in enumerate(spl):
                if i % 10 == 0:
                    new_spl += "\n"
                new_spl += token + " "
            labels.append("\n" + new_spl.strip())

        ax = sns.displot(
            df, x="Corpus", hue="Sense", stat="frequency", 
            multiple="fill", shrink=.8, legend=False
        )
        plt.legend(
            title="Sense examples", labels=labels, 
            fontsize="large", title_fontsize="20",
            bbox_to_anchor=(1.1, 1.2)
        )
        plt.title(
            target, fontsize=30, 
            verticalalignment="bottom", fontweight="bold"
        )
        ax.savefig(f"{png_path}/{idx}_{target}.png")
        ax.savefig(
            f"{eps_path}/{idx}_{target}.eps", 
            format="eps",
            dpi=1500
        )
        plt.close(ax.fig)

    

