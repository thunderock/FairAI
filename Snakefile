from os.path import join as j

DIMS = [25, 100]
WIKI = 'wiki'
EMBEDDINGS = 'embeddings'
OUTPUT = 'output'
WORD2VEC = 'word2vec'
GLOVE = 'glove'
DATASETS = [WIKI]
DATA_SRC = {
    WIKI: '../simplewiki-20171103-pages-articles-multistream.xml.bz2',
    EMBEDDINGS: 'embeddings',
    OUTPUT: 'data'
}

embeddings_params = {
    "threads": 55,
    "dim": 100,
    "embedding_dir": DATA_SRC[EMBEDDINGS],
    "output_dir": DATA_SRC[OUTPUT],
    "window_size": 8,
    "min_count": 10,
    "corpus_id": 0,
}

VOCAB_FILE = j("{embedding_dir}",'vocab-C0-V{min_count}.txt')
EMBEDDING_FILE = j("{embedding_dir}", 'vectors-C0-V{min_count}-W{window_size}-D{dim}-R0.05-E50-S1.bin'),
COOC_PATH = j("{embedding_dir}",'cooc-C0-V{min_count}-W{window_size}.bin')
SCORES_OUTPUT = j("{output_dir}", 'weat_scores_{dim}.npy')

rule calculate_glove_weat_scores_100:
    input:
        dataset = DATA_SRC[WIKI],
        vocab_file = expand(VOCAB_FILE, **embeddings_params),
        embedding_file = expand(EMBEDDING_FILE, **embeddings_params),
        cooc_path = expand(COOC_PATH, **embeddings_params),
    threads: 55
    output:
        out = expand(SCORES_OUTPUT, **embeddings_params)
    shell:
        "python driver_torch.py --dataset_type={WIKI} --dataset_file={input.dataset} --model_name={GLOVE} --dim=100 --outfile={output.out} --threads={threads}"

rule train_biased_word2vec_100:
    input:
        biased_iids = "dataset_100.pkl",
        dataset = DATA_SRC[WIKI]
    output:
        out = "biased_word2vec_100.bin"
    threads: 1
    run:
        # should I write code here instead of calling driver
        import pickle as pkl
        from utils.dataset import Dataset
        from models.word2vec import Word2Vec
        iids = pkl.load(open(input.biased_iids, 'rb'))
        ds = Dataset(input.dataset)
        model = Word2Vec(load=False, window_size=embeddings_params['window_size'],
            min_count=embeddings_params['min_count'], dim=embeddings_params['dim'])
        dataset = [ds.lines[iid] for iid in iids]
        # TODO (ashutiwa): remove iid as mandatory parameter from base class
        model.fit(iid=None, dataset=dataset)
        model.save(output.out)

rule train_fairness_aware_word2vec:
    input:
        dataset = DATA_SRC[WIKI],
        biased_iids = "dataset.pkl",
        biased_word2vec_100 = "biased_word2vec_100.bin"
    output:
        out = "fairness_aware_word2vec.bin"
    run:
        import pickle as pkl
        from models.word2vec import Word2Vec

