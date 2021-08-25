# Making History Count - Information Retrieval Project
Project repository for the Information Retrieval course of Università degli Studi di Milano (UNIMI), academic year 2020/2021.

Install with:

	pip install -e .
	
	
## Task 1: Semantic Shifts
Download fastText [wiki-news-300d-1M.vec](https://fasttext.cc/docs/en/english-vectors.html) and [HistoFast300D.zip](https://github.com/dhfbk/Histo) from the Histo repository. 

Convert the Histo embeddings in binary format. The ``save_word2vec_format`` function in ``irproject/semantic_shifts.py`` can be used for this purpose.

Download BERT's fine-tuned model [bert-semeval2020](https://drive.google.com/file/d/1LiUqP5cao3gsQMNBCceioQ6Q7vYxkPTo/view?usp=sharing) and tokenizer [bert-semeval2020-tokenizer](https://drive.google.com/file/d/1FRoNBH1G9ZDtesLwAl4nlifbqQtsQCE4/view?usp=sharing).

Download the SemEval-2020 Task 1 English data set from [here](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/). You should have a ``targets.txt``, ``ccoha1.txt`` from the ``tokens`` directory (to be renamed ``corpus_old.txt``), ``ccoha2.txt`` from the ``tokens`` directory (to be renamed ``corpus_new.txt``).

The directory tree for this task should look like the following:

    project/
        datasets/
            semeval2020/
                truth/
                    graded.txt
                corpus_new.txt
                corpus_old.txt
                targets.txt
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

### Notebooks            

The ``semantic_shifts_fine_tuning.ipynb`` notebook was used to fine-tune BERT. The ``semantic_shifts.ipynb`` notebook was used to compute the results.

### Results

The results are stored in the ``results/semantic_shifts`` directory. For the first two methods implemented in the project, the results are collected in the ``orthogonal_procrustes.json`` and ``nearest_neighbors.json`` files. These results contain the top 100 words with most semantic shift, sorted by semantic shift amount, and they can be easily interpreted. For the third method, the ``jensen_shannon.json`` file provides a detailed description of the results, but can be difficult to read. For this purpose, we also provide images in the ``results/semantic_shifts/jensen_shannon/png`` directory. In this directory, the files are sorted from the word with the most semantic shift to the word with the least semantic shift by way of the numbered prefix. 

## Task 2: Historical Event Extraction

Download the RAMS 1.0b data set [here](https://nlp.jhu.edu/rams/) and our Wikipedia data set [here](https://drive.google.com/file/d/1exT7OK-7ViONUE6JPeBRD8vTy6ObusvS/view?usp=sharing).

Download the training checkpoints for our [Multi-Task Learning Model](https://drive.google.com/file/d/10UHxbZl8B5qk3Kpok8c0rThxFBUljWoH/view?usp=sharing), [Event Model](https://drive.google.com/file/d/16UcLQnaCUx_oqyDd3rspFLiIqGSdWwnt/view?usp=sharing), [EventGen Model](https://drive.google.com/file/d/1gM_4lEFmciSJhF-W1Ka-MsRS3HysTG8L/view?usp=sharing) and [Argument Model](https://drive.google.com/file/d/1ZXv2w57mHWtis4xbsteC2WF-VnuNJoU2/view?usp=sharing).

Download the ``aida_ontology_cleaned.csv`` file from [here](https://github.com/raspberryice/gen-arg).

The directory tree for this task should look like this:

    project/
        checkpoints/
            historical_events/
                argument/
                    epoch=2-step=21986.ckpt
                event/
                    epoch=9-step=36649.ckpt
                event_gen/
                    epoch=1-step=14657.ckpt
                mtl/
                    epoch=13-step=45415.ckpt
        datasets/
            historical_events/
                wiki_dataset.json
            rams/
                raw/
                    dev.jsonlines
                    test.jsonlines
                    train.jsonlines
            aida_ontology_cleaned.csv

### Notebooks

The ``wiki_dataset_builder.ipynb`` notebook was employed to build our Wikipedia data set and can be used to retrieve additional pages. 

The ``paragraph_classification.ipynb`` notebook was used to train and evaluate our Multi-Task Learning Model and the baseline methods for paragraph classification.

The ``event_extraction.ipynb`` notebook was used to train and evaluate our models on the event and argument extraction tasks.

### Results

The results for this task are stored as JSON files in the ``results/historical_events`` subdirectories. In the ``wiki_dataset`` subdirectory we stored predictions on the training set of our Wikipedia data set. Some noteworthy examples of our models capabilities on the Wikipedia data set can be found in our slides ``latex/slides/slides.pdf``.