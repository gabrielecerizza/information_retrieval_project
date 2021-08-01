# Making History Count - Information Retrieval Project
Project repository for the Information Retrieval course in University of Milan (UNIMI), academic year 2020/2021.

Install with:

	pip install -e .
	
	
## Task 1: Semantic Shifts
Download *fastText* [wiki-news-300d-1M.vec](https://fasttext.cc/docs/en/english-vectors.html) and [cc.en.300.bin](https://fasttext.cc/docs/en/crawl-vectors.html). Also download [HistoFast300D.zip](https://github.com/dhfbk/Histo) from the *Histo* repository. 

Convert the *Histo* embeddings in binary format. The ``save_word2vec_format`` function in ``irproject/semantic_shifts.py`` can be used for this purpose.

Download *BERT*'s fine-tuned model [bert-semeval2020](https://drive.google.com/file/d/1LiUqP5cao3gsQMNBCceioQ6Q7vYxkPTo/view?usp=sharing) and tokenizer [bert-semeval2020-tokenizer](https://drive.google.com/file/d/1FRoNBH1G9ZDtesLwAl4nlifbqQtsQCE4/view?usp=sharing).

Download the SemEval-2020 Task 1 English dataset from [here](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/). You should have a ``targets.txt``, ``ccoha1.txt`` from the ``tokens`` directory (to be renamed ``corpus_old.txt``), ``ccoha2.txt`` from the ``tokens`` directory (to be renamed ``corpus_new.txt``).

The directory tree for this task should look like the following:

    project/
        datasets/
            semeval2020/
                truth/
                    graded.txt
                corpus_new.txt
                corpus_old.txt
                targets.txt
            cc.en.300.bin
            histo-fast-300d.bin
            wiki-news-300d-1M.vec
        pretrained/
            bert-semeval2020/
                config.json
                pytorch_model.bin
                training_args.bin
            bert-semeval2020-tokenizer/
                special_tokens_map.json
                tokenizer.json
                tokenizer_config.json
                vocab.txt
            
The results are stored in the ``results/semantic_shifts`` directory. For the first two methods implemented in the project, the results are collected in the ``orthogonal_procrustes.json`` and ``nearest_neighbors.json`` files. These results can be easily interpreted. For the third method, the ``jensen_shannon.json`` file provides a detailed description of the results, but can be difficult to read. For this purpose, we also provide images in the ``results/semantic_shifts/jensen_shannon/png`` directory. In this directory, the files are sorted from the word with the most semantic shift to the word with the least semantic shift by way of the numbered prefix.  