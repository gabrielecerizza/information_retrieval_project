# Making History Count - Information Retrieval Project
Project repository for the Information Retrieval course of Università degli Studi di Milano (UNIMI), academic year 2020/2021.

Install with:

	pip install -e .
	
	
## Task 1: Semantic Shifts
Download fastText [wiki-news-300d-1M.vec](https://fasttext.cc/docs/en/english-vectors.html) and [HistoFast300D.zip](https://github.com/dhfbk/Histo) from the Histo repository. 

Convert the Histo embeddings in binary format. The ``save_word2vec_format`` function in ``irproject/semantic_shifts.py`` can be used for this purpose.

Download the SemEval-2020 Task 1 English data set from [here](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/). You should have a ``targets.txt``, ``ccoha1.txt`` from the ``tokens`` directory (to be renamed ``corpus_old.txt``), ``ccoha2.txt`` from the ``tokens`` directory (to be renamed ``corpus_new.txt``).

### Notebooks            

The ``semantic_shifts_fine_tuning.ipynb`` notebook was used to fine-tune BERT. The ``semantic_shifts.ipynb`` notebook was used to compute the results.

### Results

The results are stored in the ``results/semantic_shifts`` directory. For the first two methods implemented in the project, the results are collected in the ``orthogonal_procrustes.json`` and ``nearest_neighbors.json`` files. These results contain the top 100 words with most semantic shift, sorted by semantic shift amount, and they can be easily interpreted. For the third method, the ``jensen_shannon.json`` file provides a detailed description of the results, but can be difficult to read. For this purpose, we also provide images in the ``results/semantic_shifts/jensen_shannon/png`` directory. In this directory, the files are sorted from the word with the most semantic shift to the word with the least semantic shift by way of the numbered prefix. 

## Task 2: Historical Event Extraction

Download the RAMS 1.0b data set [here](https://nlp.jhu.edu/rams/).

Download the ``aida_ontology_cleaned.csv`` file from [here](https://github.com/raspberryice/gen-arg).

### Notebooks

The ``wiki_dataset_builder.ipynb`` notebook was employed to build our Wikipedia data set and can be used to retrieve additional pages. 

The ``paragraph_classification.ipynb`` notebook was used to train and evaluate our Multi-Task Learning Model and the baseline methods for paragraph classification.

The ``event_extraction.ipynb`` notebook was used to train and evaluate our models on the event and argument extraction tasks.

### Results

The results for this task are stored as JSON files in the ``results/historical_events`` subdirectories. In the ``wiki_dataset`` subdirectory we stored predictions on our Wikipedia data set. Some noteworthy examples of our models capabilities on the Wikipedia data set can be found in our slides (``latex/slides/slides.pdf``).
