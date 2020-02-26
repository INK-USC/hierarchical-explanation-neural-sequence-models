# Hierarchical Explanations for Neural Sequence Model Predictions

This repo include implementation of SOC and SCD algorithm, scripts for visualization and evaluation. **The code clean-up is still in progress.**

Paper: [Towards Hierarchical Importance Attribution: Explaining Compositional Semantics for Neural Sequence Models](https://openreview.net/pdf?id=BkxRRkSKwr), ICLR 2020.

## Installation
```shell script
conda create -n hiexpl-env python==3.7.4
conda activate hiexpl-env
# modify CUDA version as yours
conda install pytorch=0.4.1 cuda90 -c pytorch
pip install nltk numpy scikit-learn scikit-image matplotlib torchtext
# requirements from pytorch-transformers
pip install tokenizers==0.0.11 boto3 filelock requests tqdm regex sentencepiece sacremoses
```

## Pipeline
### SST-2 (LSTM)

Train a LSTM classifier. The SST-2 dataset will be downloaded automatically.

```shell script
mkdir models
export model_path=models/sst_lstm
python train.py --task sst --save_path models/${model_path} --no_subtrees --lr 0.0005
``` 

Pretrain a language model on the training set.
```shell script
export lm_path=models/sst_lstm_lm
python -m lm.lm_train --task sst --save_path models/${lm_path} --no_subtrees --lr 0.0002
```

Use SOC/SCD to interpret first 50 predictions on dev set. `nb_range=10` and `sample_n=20` recommended. 
```shell script
export algo=soc # or scd
export exp_name=.sst_demo
mkdir sst
mkdir sst/results
python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 50 --exp_name ${exp_name} --task sst --explain_model lstm --nb_range 10 --sample_n 20
```
Check `outputs/sst/soc_results/` or `outputs/sst/scd_results` for explanation outputs. Each line contains one instance, where score attribution for each word/phrase is tab-separated. For example:

```
it 0.142656	 's 0.192409	 the 0.175471	 best 0.829247	 film 0.095305	 best film 0.805854	 the best film 1.004583 ...
```

Or use `--agg` flag to automatically construct hierarchical explanations,  without need for ground truth parsing trees.

```shell script
python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 50 --exp_name ${exp_name} --task sst --explain_model lstm --nb_range 10 --sample_n 20 --agg
```

The output can be read by `visualize.py` to generate visualizations.

### SST-2 (BERT)
Download SST-2 dataset from https://gluebenchmark.com/ and unzip at `bert/glue_data`. Then finetune the BERT model to build a classifier. 

```shell script
python -m bert.run_classifier \
  --task_name SST-2 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir bert/glue_data/SST-2 \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir bert/models_sst 
```

Then train a language model on BERT-tokenized inputs, and run explanations. Simply add the `--use_bert_tokenizer` and `--explain_model bert` flag to all the experiments above for LSTM.

Note that you need to filter out subtrees in train.tsv if you are interested in evaluating explanations.

## Evaluating explanations

To evaluate word level explanation, a BOW linear classifier is required.
```shell script
python -m nns.linear_model --task sst --save_path models/${model_path}
```

For evaluation of phrase level explanation, you also need to download the original SST dataset.
```shell script
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip ./.data/stanfordSentimentTreebank.zip -d ./.data/
mv ./.data/stanfordSentimentTreebank ./.data/sst_raw
```

Then run the evaluation script:
```shell script
python eval_explanations.py --eval_file outputs/soc${exp_name}.txt
```


## Contact

If you have any questions about the paper or the code, please feel free to contact Xisen Jin (xisenjin usc edu).

