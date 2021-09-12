# DAMS

Pytorch implementation of the EMNLP-2021 paper: [Low-Resource Dialogue Summarization with Domain-Agnostic Multi-Source Pretraining](https://arxiv.org/abs/2109.04080).

## Requirements

* Python 3.7.10
 
* pytorch 1.7.0+cu11.0

* py-rouge 1.1

* transformers 4.0.0

* multiprocess 0.70.11.1

* tensorboardX 2.1

* torchtext 0.4.0

* nltk 3.6.2

## Environment

* RTX 3090 GPU

* CUDA 11.1

## Data

All the datastes used in our work are available at [Google Drive](https://drive.google.com/file/d/1SpTkDukxxyy01uN6Gct1023cjT_0Y24h/view?usp=sharing) or [Baidu Pan  (extract code: wwsd)](https://pan.baidu.com/s/1tfOOL9ci989L1FM5CSCQPw), including the multi-source pretraining data and the dialogue summary data.

## Usage

* Download BERT checkpoints [here](https://huggingface.co/bert-base-uncased/tree/bb3c1c3256d2598217df9889a14a2e811587891d) and put BERT checkpoints into the directory **bert** like this:

	```
	--- bert
	  |
	  |--- bert_base_uncased
	     |
	     |--- config.json
	     |
	     |--- pytorch_model.bin
	     |
	     |--- vocab.txt
	```

* Pre-process dialogue summary datasets (e.g., the SAMSum training data).

    ```
    PYTHONPATH=. python ./src/preprocess.py -type train -raw_path json_data/samsum -save_path torch_data/samsum -log_file logs/json_to_data_samsum.log -truncated -n_cpus 4
    ```

* Pre-process multi-source pretraining datasets and mix them up.

    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path json_data -save_path torch_data/all -log_file logs/json_to_data.log -truncated -n_cpus 40 -mix_up
    ```

* Pretrain DAMS on the multi-source datasets.

    ```
    PYTHONPATH=. python ./src/main.py -mode train -data_path torch_data/all/data -model_path models/pretrain -log_file logs/pretrain.log -sep_optim -pretrain -visible_gpus 0,1 -pretrain_steps 250000 -port 10000
    ```

* Fine-tune DAMS on the SAMSum training set.
    ```
    PYTHONPATH=. python ./src/main.py -mode train -data_path torch_data/samsum/samsum -model_path models/samsum -log_file logs/samsum.train.log -visible_gpus 0 -warmup_steps 1000 -lr 0.001 -train_from models/pretrain/model_step_250000.pt -train_from_ignore_optim -train_steps 50000
    ```

* Validate DAMS on the SAMSum validation set.

    ```
    PYTHONPATH=. python ./src/main.py -mode validate -data_path torch_data/samsum/samsum -log_file logs/samsum.val.log -val_all -alpha 0.95 -model_path models/samsum -result_path results/samsum/samsum -visible_gpus 0 -min_length 15 -beam_size 3 -test_batch_ex_size 50
    ```

* Test DAMS.

    * Zero-shot test on the SAMSum test set using the pretrained model.

    ```
    PYTHONPATH=. python ./src/main.py -mode test -data_path torch_data/samsum/samsum -log_file logs/samsum.test.log -alpha 0.95 -test_from models/pretrain/model_step_250000.pt -result_path results/samsum/samsum -visible_gpus 0 -min_length 15 -beam_size 3 -test_batch_ex_size 50
    ```

    * Regular test on the SAMSum test set using the best validated model.

    ```
    PYTHONPATH=. python ./src/main.py -mode test -data_path torch_data/samsum/samsum -log_file logs/samsum.test.log -alpha 0.95 -test_from models/samsum/model_step_xxx.pt -result_path results/samsum/samsum -visible_gpus 0 -min_length 15 -beam_size 3 -test_batch_ex_size 50
    ```

    * Transfer to the ADSC test set.
    ```
    PYTHONPATH=. python ./src/main.py -mode test -data_path torch_data/adsc/adsc -log_file logs/adsc.test.log -alpha 0.95 -test_from models/samsum/model_step_xxx.pt -result_path results/adsc/adsc -visible_gpus 0 -min_length 100 -beam_size 3 -test_batch_ex_size 50
    ```

## Citation

	@ARTICLE{2021arXiv210904080Z,
             author = {Zou, Yicheng and Zhu, Bolin and Hu, Xingwu and Gui, Tao and Zhang, Qi},
             title = "{Low-Resource Dialogue Summarization with Domain-Agnostic Multi-Source Pretraining}",
             journal = {arXiv e-prints},
             keywords = {Computer Science - Computation and Language},
             year = 2021,
             month = sep,
             eid = {arXiv:2109.04080},
             pages = {arXiv:2109.04080},
             archivePrefix = {arXiv},
             eprint = {2109.04080},
             primaryClass = {cs.CL},
             adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210904080Z}
            }

