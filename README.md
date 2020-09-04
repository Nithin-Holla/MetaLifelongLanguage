# MetaLifelongLanguage

## Getting started

- Clone the repository: `git clone git@github.com:Nithin-Holla/MetaLifelongLanguage.git`.
- Create a virtual environment.
- Install the required packages: `pip install -r MetaLifelongLanguage/requirements.txt`.

## Downloading the data

- Create a directory for storing the data: `mkdir data`.
- Navigate to the data directory: `cd data`.
- Download the five datasets for text classification from [here](https://tinyurl.com/y89zdadp) and unzip them in this directory.
- Make a new directory for lifelong relation extraction: `mkdir LifelongFewRel`.
- Download the files using these commands:
    - `wget https://raw.githubusercontent.com/hongwang600/Lifelong_Relation_Detection/master/data/relation_name.txt`
    - `wget https://raw.githubusercontent.com/hongwang600/Lifelong_Relation_Detection/master/data/training_data.txt`
    - `wget https://raw.githubusercontent.com/hongwang600/Lifelong_Relation_Detection/master/data/val_data.txt`
- Navigate back: `cd ../..`.
- The directory tree should like this:
<pre>
.
├── MetaLifelongLanguage
├── data
│   ├── ag_news_csv
│   │   ├── classes.txt
│   │   ├── readme.txt
│   │   ├── test.csv
│   │   └── train.csv
│   ├── amazon_review_full_csv
│   │   ├── readme.txt
│   │   ├── test.csv
│   │   └── train.csv
│   ├── dbpedia_csv
│   │   ├── classes.txt
│   │   ├── readme.txt
│   │   ├── test.csv
│   │   └── train.csv
│   ├── yahoo_answers_csv
│   │   ├── classes.txt
│   │   ├── readme.txt
│   │   ├── test.csv
│   │   └── train.csv
│   ├── yelp_review_full_csv
│   │   ├── readme.txt
│   │   ├── test.csv
│   │   └── train.csv
│   ├── LifelongFewRel
│   │   ├── relation_name.txt
│   │   ├── training_data.txt
│   │   ├── val_data.txt
</pre>


## Text classification

`train_text_cls.py` contains the code for training and evaluation on the lifelong text classification benchmark. The usage is:
```
python train_text_cls.py [-h] --order ORDER [--n_epochs N_EPOCHS] [--lr LR]
                         [--inner_lr INNER_LR] [--meta_lr META_LR]
                         [--model MODEL] [--learner LEARNER]
                         [--n_episodes N_EPISODES]
                         [--mini_batch_size MINI_BATCH_SIZE]
                         [--updates UPDATES] [--write_prob WRITE_PROB]
                         [--max_length MAX_LENGTH] [--seed SEED]
                         [--replay_rate REPLAY_RATE]
                         [--replay_every REPLAY_EVERY]

optional arguments:
  -h, --help            show this help message and exit
  --order ORDER         Order of datasets
  --n_epochs N_EPOCHS   Number of epochs (only for baseline)
  --lr LR               Learning rate (only for baseline)
  --inner_lr INNER_LR   Inner-loop learning rate
  --meta_lr META_LR     Meta learning rate
  --model MODEL         Name of the model
  --learner LEARNER     Learner method
  --n_episodes N_EPISODES
                        Number of meta-training episodes
  --mini_batch_size MINI_BATCH_SIZE
                        Batch size of data points within an episode
  --updates UPDATES     Number of inner-loop updates
  --write_prob WRITE_PROB
                        Write probability for buffer memory
  --max_length MAX_LENGTH
                        Maximum sequence length for the input
  --seed SEED           Random seed
  --replay_rate REPLAY_RATE
                        Replay rate from memory
  --replay_every REPLAY_EVERY
                        Number of data points between replay
```

## Relation extraction

`train_rel.py` contains the code for training and evaluating on the lifelong relation extraction benchmark. The usage is:
```
python train_rel.py [-h] [--n_epochs N_EPOCHS] [--lr LR] [--inner_lr INNER_LR]
                    [--meta_lr META_LR] [--model MODEL] [--learner LEARNER]
                    [--n_episodes N_EPISODES]
                    [--mini_batch_size MINI_BATCH_SIZE] [--updates UPDATES]
                    [--write_prob WRITE_PROB] [--max_length MAX_LENGTH]
                    [--seed SEED] [--replay_rate REPLAY_RATE] [--order ORDER]
                    [--num_clusters NUM_CLUSTERS]
                    [--replay_every REPLAY_EVERY]

optional arguments:
  -h, --help            show this help message and exit
  --n_epochs N_EPOCHS   Number of epochs (only for baseline)
  --lr LR               Learning rate (only for baseline)
  --inner_lr INNER_LR   Inner-loop learning rate
  --meta_lr META_LR     Meta learning rate
  --model MODEL         Name of the model
  --learner LEARNER     Learner method
  --n_episodes N_EPISODES
                        Number of meta-training episodes
  --mini_batch_size MINI_BATCH_SIZE
                        Batch size of data points within an episode
  --updates UPDATES     Number of inner-loop updates
  --write_prob WRITE_PROB
                        Write probability for buffer memory
  --max_length MAX_LENGTH
                        Maximum sequence length for the input
  --seed SEED           Random seed
  --replay_rate REPLAY_RATE
                        Replay rate from memory
  --order ORDER         Number of task orders to run for
  --num_clusters NUM_CLUSTERS
                        Number of clusters to take
  --replay_every REPLAY_EVERY
                        Number of data points between replay
```
