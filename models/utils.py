import gc
import json
import torch
import sqlparse
import argparse
import warnings
import tiktoken
import numpy as np
import pandas as pd
import torch.nn as nn
from os import cpu_count
from copy import deepcopy
from os.path import exists
from datetime import datetime
from dotenv import dotenv_values
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from openai import OpenAI, NotGiven, NOT_GIVEN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score

from .multiclass import MULTICLASS_RUN_MODES, ERLoss
from .binary import BINARY_RUN_MODES

warnings.filterwarnings('ignore', category=FutureWarning)

def prepare_data(data: pd.DataFrame, filepath: str, filename: str, binary_model: bool, dims: int|NotGiven = NOT_GIVEN, augment: bool = False) -> tuple:
    """
    helper function to format and output data for modelling

    Args:
        data (pd.DataFrame): query data
        filepath (str): filepath
        filename (str): filename
        binary_model (bool): indicator for training binary classification or multiclass classification task
        dims (int | NotGiven, optional): number of dimensions to embed queries into. Defaults to NOT_GIVEN.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: a tuple of torch tensors
    """    
    model_df: pd.DataFrame = data.copy()

    model_df = model_df.drop(columns=['runtime_list', 'plan_tree', 'sd_runtime'])
    model_df = model_df.explode(column='hint_list')
    model_df = model_df.sort_values(by=['filename', 'hint_list'])
    model_df = model_df.groupby(by=['filename', 'sql'], as_index=False).agg({
        'hint_list': lambda x: x.tolist(), 
        'mean_runtime': lambda x: x.tolist()
    })
    model_df['opt_l'] = model_df.mean_runtime.apply(min)
    
    if augment:
        model_df, syntaxB, syntaxC = generate_embeddings(model_df, filepath, filename, dims, augment)
    else:
        model_df = generate_embeddings(model_df, filepath, filename, dims, augment)
    model_df = model_df.drop(columns=['filename', 'sql', 'hint_list'])

    X = torch.stack(model_df.features.apply(lambda x: torch.Tensor(x)).tolist())
    hint_l = torch.stack(model_df.mean_runtime.apply(lambda x: torch.Tensor(x)).tolist())
    opt_l = torch.stack(model_df.opt_l.apply(lambda x: torch.Tensor([x]).repeat(hint_l.size(1))).tolist())
    if binary_model:
        BENCHMARK_IDX = 0
        LONGTAIL_IDX = 26
        binary_l = (hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]).type(torch.float)
    else:
        binary_l = (hint_l == opt_l).type(torch.float)
    
    if augment:
        return X, hint_l, opt_l, binary_l, syntaxB, syntaxC
    else:
        return X, hint_l, opt_l, binary_l


def generate_embeddings(data: pd.DataFrame, filepath: str, filename: str, dims: int|NotGiven = NOT_GIVEN, augment: bool = False) -> pd.DataFrame:
    """
    helper function that generates embeddings from LLM

    Args:
        data (pd.DataFrame): query data
        filepath (str): filepath
        filename (str): filename
        dims (int | NotGiven, optional): number of dimensions to embed queries into. Defaults to NOT_GIVEN.

    """    
    MAX_TOKENS = 8191
    MAX_LEN = 2048

    embeddings = None
    f_path = f"{filepath}/{filename}{'_'+str(dims) if dims != NOT_GIVEN else ''}.json"
    subset_df = data[['filename','sql']].drop_duplicates()
    
    if exists(f_path):
        with open(f_path, mode='r', encoding='utf-8') as f:
            embeddings = json.load(f)
    else:
        embeddings = dict()

        # load api key from .env
        key = dotenv_values('.env')

        # start client
        client = OpenAI(api_key=key['OPENAI_KEY'])
        model = 'text-embedding-3-large'

        inputs = subset_df.sql.to_list()

        last_idx = 0
        input_idx = 0
        while input_idx < len(inputs):
            input = list()
            input_len = 0
            tokens = 0

            while tokens < MAX_TOKENS and input_len < MAX_LEN and input_idx < len(inputs):
                cand_tokens = count_tokens(inputs[input_idx], model)

                if (tokens+cand_tokens) < MAX_TOKENS:
                    input.append(inputs[input_idx])
                    tokens += cand_tokens
                    input_len += 1
                    input_idx += 1
                else: 
                    break

            result = client.embeddings.create(
                input=input,
                model=model,
                dimensions=dims,
            )
            embeddings.update(dict(zip(subset_df.filename.to_list()[last_idx:input_idx], result.model_dump()['data'])))
            last_idx = input_idx

        with open(f_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=4)
    
    data['features'] = data.filename.apply(lambda x: embeddings[x]['embedding'])

    if not augment:
        return data
    else:
        syntaxb_path = f"{filepath}/{filename.replace('syntaxA', 'syntaxB')}{'_'+str(dims) if dims != NOT_GIVEN else ''}.json"
        syntaxc_path = f"{filepath}/{filename.replace('syntaxA', 'syntaxC')}{'_'+str(dims) if dims != NOT_GIVEN else ''}.json"
        syntaxB_queries = syntaxC_queries = None

        for path in [syntaxb_path, syntaxc_path]:
            if exists(path):
                with open(path, mode='r', encoding='utf-8') as f:
                    embeddings = json.load(f)
            else:
                if 'client' not in locals() or model not in locals():
                        # load api key from .env
                        key = dotenv_values('.env')

                        # start client
                        client = OpenAI(api_key=key['OPENAI_KEY'])
                        model = 'text-embedding-3-large'
                
                inputs = subset_df.sql.apply(lambda x: sqlparse.format(x, reindent=True, use_space_around_operators=True, indent_tabs=False)) if 'syntaxB' in path else data.sql.apply(lambda x: sqlparse.format(x, reindent=True, use_space_around_operators=True, indent_tabs=True))
                last_idx = 0
                input_idx = 0
                while input_idx < len(inputs):
                    input = list()
                    input_len = 0
                    tokens = 0

                    while tokens < MAX_TOKENS and input_len < MAX_LEN and input_idx < len(inputs):
                        cand_tokens = count_tokens(inputs[input_idx], model)

                        if (tokens+cand_tokens) < MAX_TOKENS:
                            input.append(inputs[input_idx])
                            tokens += cand_tokens
                            input_len += 1
                            input_idx += 1
                        else: 
                            break

                    result = client.embeddings.create(
                        input=input,
                        model=model,
                        dimensions=dims,
                    )
                    embeddings.update(dict(zip(subset_df.filename.to_list()[last_idx:input_idx], result.model_dump()['data'])))
                    last_idx = input_idx

                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(embeddings, f, ensure_ascii=False, indent=4)

            if 'syntaxB' in path:
                syntaxB_queries=torch.Tensor(subset_df.filename.apply(lambda x: embeddings[x]['embedding']))
            else:
                syntaxC_queries=torch.Tensor(subset_df.filename.apply(lambda x: embeddings[x]['embedding']))

        return data, syntaxB_queries, syntaxC_queries
    

def count_tokens(input: str, model: str) -> int:
    """
    counts the number of tokens that would be parsed out of a given string given a LLM

    Args:
        input (str): the string to be tokenized
        model (str): the model underwhich the input is to be tokenized

    Returns:
        int: number of toeks
    """    
    encodings = {
        'gpt-4o': 'o200k_base',
        'gpt-4': 'cl100k_base',
        'gpt-3.5-turbo': 'cl100k_base', 
        'davinci-002':'cl100k_base', 
        'babbage-002':'cl100k_base', 
        'text-embedding-ada-002':'cl100k_base', 
        'text-embedding-3-large':'cl100k_base', 
        'text-embedding-3-small':'cl100k_base', 
    }

    encoding = tiktoken.get_encoding(encoding_name=encodings.get(model))
    return len(encoding.encode(input))

def train_model(
        exp: int|None,
        binary_model: bool,
        X: torch.Tensor, 
        hint_latencies: torch.Tensor,
        opt_latencies: torch.Tensor, 
        targets: torch.Tensor, 
        components: int, 
        post_process: bool,
        neural_model: nn.Module,
        loss_model: nn.Module,
        epochs: int,
        device: str,
        k: int = 5,
    ) -> dict:
    """
    a worker function that facilitates the training of neural network models

    Args:
        exp (int | None): experiment number
        binary_model (bool): indicator for training binary classification or multiclass classification task
        X (torch.Tensor): input data (m x p)
        hint_latencies (torch.Tensor): hint latencies (m x 49)
        opt_latencies (torch.Tensor): optimal hint latencies (m x 49)
        targets (torch.Tensor): optimal hints binary encoded (m x 49)
        components (int): the number of principal components to be used as input
        post_process (bool): indicator to standardize principal components before training
        neural_model (nn.Module): a torch model instance
        loss_model (nn.Module): a torch loss function
        epochs (int): the number epochs to train the model for
        device (str): gpu or cpu
        k (int, optional): the number of cross-validation folds to perform. Defaults to 5.

    Returns:
        dict: summary of the model performance and other experiment metadata
    """    
    print(f'[{datetime.now().isoformat()}] {type(neural_model).__name__}{f"" if exp is None else " [Experiment #" + str(exp) + "]"} - Beginning to Train on {device.upper()}')
    
    THRESHOLD = 0.5 if binary_model else None
    RANDOM_SEED = 24508
    rng = np.random.default_rng(seed=RANDOM_SEED)
    p = 0.8
    epochs = epochs
    batch_size = 64
    
    model_perf = {
        'train_loss': [],
        'model_preds_train': [],
        'model_preds_test': [],
        'model_probs_train': [],
        'model_probs_test': [],

        'model_train_accuracy': [],
        'model_test_accuracy': [],
        'apriori_train_distribution': [],
        'apriori_test_distribution': [],

        'model_train_recall': [],
        'model_test_recall': [],
        'model_train_precision': [],
        'model_test_precision': [],
        'model_train_f1score': [],
        'model_test_f1score': [],
        'model_train_auroc': [],
        'model_test_auroc': [],

        'train_model_workload': [],
        'train_opt_workload': [],
        'train_benchmark_workload': [],
        'train_apriori_workload': [],
        'test_model_workload': [],
        'test_opt_workload': [],
        'test_benchmark_workload': [],
        'test_apriori_workload': [],

        'train_model_p90': [],
        'train_opt_p90': [],
        'train_benchmark_p90': [],
        'train_apriori_p90': [],
        'test_model_p90': [],
        'test_opt_p90': [],
        'test_benchmark_p90': [],
        'test_apriori_p90': [],

        'train_model_median': [],
        'train_opt_median': [],
        'train_benchmark_median': [],
        'train_apriori_median': [],
        'test_model_median': [],
        'test_opt_median': [],
        'test_benchmark_median': [],
        'test_apriori_median': [],

    } if binary_model else {
        'train_loss': [],
        'model_probs_train': [],
        'model_probs_test': [],

        'train_model_workload': [],
        'train_opt_workload': [],
        'train_benchmark_workload': [],
        'test_model_workload': [],
        'test_opt_workload': [],
        'test_benchmark_workload': [],

        'train_model_p90': [],
        'train_opt_p90': [],
        'train_benchmark_p90': [],
        'test_model_p90': [],
        'test_opt_p90': [],
        'test_benchmark_p90': [],

        'train_model_median': [],
        'train_opt_median': [],
        'train_benchmark_median': [],
        'test_model_median': [],
        'test_opt_median': [],
        'test_benchmark_median': [],
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA())
    ])
    split_idxs = list(StratifiedShuffleSplit(n_splits=k, train_size=p, random_state=RANDOM_SEED).split(X, targets)) if binary_model else list()

    for fold in range(k):
        if binary_model:
            X_train, hint_l_train, opt_l_train, targets_l_train = X[split_idxs[fold][0],], hint_latencies[split_idxs[fold][0],], opt_latencies[split_idxs[fold][0],], targets[split_idxs[fold][0],]
            X_test, hint_l_test, opt_l_test, targets_l_test = X[split_idxs[fold][1],], hint_latencies[split_idxs[fold][1],], opt_latencies[split_idxs[fold][1],], targets[split_idxs[fold][1],]
        else:
            train_split = round(X.size(0)*p)
            shuffle_idx = rng.permutation(X.size(0))
            X_train, hint_l_train, opt_l_train, targets_l_train = X[shuffle_idx[:train_split],], hint_latencies[shuffle_idx[:train_split],], opt_latencies[shuffle_idx[:train_split],], targets[shuffle_idx[:train_split],]
            X_test, hint_l_test, opt_l_test, targets_l_test = X[shuffle_idx[train_split:],], hint_latencies[shuffle_idx[train_split:],], opt_latencies[shuffle_idx[train_split:],], targets[shuffle_idx[train_split:],]

        X_train = torch.Tensor(pipeline.fit_transform(X_train))[:,:components]
        X_test = torch.Tensor(pipeline.transform(X_test))[:,:components]

        if post_process:
            scaler = StandardScaler()
            X_train = torch.Tensor(scaler.fit_transform(X_train))
            X_test = torch.Tensor(scaler.transform(X_test))

        X_train, hint_l_train, opt_l_train, targets_l_train = X_train.to(device), hint_l_train.to(device), opt_l_train.to(device), targets_l_train.to(device)
        X_test, hint_l_test, opt_l_test, targets_l_test =  X_test.to(device), hint_l_test.to(device), opt_l_test.to(device), targets_l_test.to(device)
        model = deepcopy(neural_model).to(device)
        loss_func = deepcopy(loss_model).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        dataset = torch.utils.data.TensorDataset(X_train, hint_l_train, opt_l_train, targets_l_train)
        batches = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model_loss = list()

        if binary_model:
            loss_func.pos_weight = (targets_l_train == 0).sum() / targets_l_train.sum()

        for epoch in range(epochs):
            train_loss = 0
            N = 0
            
            for i, batch in enumerate(batches):
                inputs, hint_l, opt_l, targets_l = batch
        
                N += inputs.size(0)
                optimizer.zero_grad()

                probs = model(inputs)

                loss = loss_func(probs, targets_l) if binary_model else loss_func(probs, hint_l, opt_l)
                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            model_loss.append(train_loss/N)

        with torch.no_grad():
            if binary_model:
                logits_train, logits_test = model(X_train).to('cpu'), model(X_test).to('cpu')
                probs_train, probs_test = nn.Sigmoid()(logits_train), nn.Sigmoid()(logits_test)
            else:
                probs_train, probs_test = model(X_train).to('cpu'), model(X_test).to('cpu')

        model_perf['model_probs_train'].append(probs_train.to('cpu'))
        model_perf['model_probs_test'].append(probs_test.to('cpu'))
        if binary_model:
            benchmark_const = torch.LongTensor([0])
            longtail_const = torch.LongTensor([26])

            train_model_hints =  hint_l_train.gather(1, torch.where(probs_train > THRESHOLD, longtail_const, benchmark_const).view(-1,1))
            test_model_hints = hint_l_test.gather(1, torch.where(probs_test > THRESHOLD, longtail_const, benchmark_const).view(-1,1))
            train_apriori_hints = hint_l_train.gather(1, torch.where(targets_l_train > THRESHOLD, longtail_const, benchmark_const).view(-1,1))
            test_apriori_hints = hint_l_test.gather(1, torch.where(targets_l_test > THRESHOLD, longtail_const, benchmark_const).view(-1,1))

            preds_train, preds_test = (probs_train > THRESHOLD).type(torch.float), (probs_test > THRESHOLD).type(torch.float)
            model_perf['model_preds_train'].append(preds_train.to('cpu'))
            model_perf['model_preds_test'].append(preds_test.to('cpu'))

            model_perf['model_train_accuracy'].append(((preds_train == targets_l_train).sum()/targets_l_train.shape[0]).to('cpu'))
            model_perf['model_test_accuracy'].append(((preds_test == targets_l_test).sum()/targets_l_test.shape[0]).to('cpu'))
            model_perf['apriori_train_distribution'].append(((targets_l_train.shape[0] - targets_l_train.sum())/targets_l_train.shape[0]).to('cpu'))
            model_perf['apriori_test_distribution'].append(((targets_l_test.shape[0] - targets_l_test.sum())/targets_l_test.shape[0]).to('cpu'))
            model_perf['model_train_recall'].append(recall_score(targets_l_train,preds_train))
            model_perf['model_test_recall'].append(recall_score(targets_l_test,preds_test))
            model_perf['model_train_precision'].append(precision_score(targets_l_train,preds_train))
            model_perf['model_test_precision'].append(precision_score(targets_l_test,preds_test))
            model_perf['model_train_f1score'].append(f1_score(targets_l_train,preds_train))
            model_perf['model_test_f1score'].append(f1_score(targets_l_test,preds_test))
            model_perf['model_train_auroc'].append(roc_auc_score(targets_l_train,probs_train))
            model_perf['model_test_auroc'].append(roc_auc_score(targets_l_test,probs_test))

            train_model_workload = train_model_hints.sum()
            train_opt_workload = opt_l_train.mean(dim=1).sum()
            train_benchmark_workload = hint_l_train[:,0].sum()
            train_apriori_workload = train_apriori_hints.sum()
            test_model_workload = test_model_hints.sum()
            test_opt_workload = opt_l_test.mean(dim=1).sum()
            test_benchmark_workload = hint_l_test[:,0].sum()
            test_apriori_workload = test_apriori_hints.sum()
            model_perf['train_model_workload'].append(train_model_workload.to('cpu'))
            model_perf['train_opt_workload'].append(train_opt_workload.to('cpu'))
            model_perf['train_benchmark_workload'].append(train_benchmark_workload.to('cpu'))
            model_perf['train_apriori_workload'].append(train_apriori_workload.to('cpu'))
            model_perf['test_model_workload'].append(test_model_workload.to('cpu'))
            model_perf['test_opt_workload'].append(test_opt_workload.to('cpu'))
            model_perf['test_benchmark_workload'].append(test_benchmark_workload.to('cpu'))
            model_perf['test_apriori_workload'].append(test_apriori_workload.to('cpu'))

            train_model_p90 = train_model_hints.quantile(0.90)
            train_opt_p90 = opt_l_train.mean(dim=1).quantile(0.90)
            train_benchmark_p90 = hint_l_train[:,0].quantile(0.90)
            train_apriori_p90 = train_apriori_hints.quantile(0.90)
            test_model_p90 = test_model_hints.quantile(0.90)
            test_opt_p90 = opt_l_test.mean(dim=1).quantile(0.90)
            test_benchmark_p90 = hint_l_test[:,0].quantile(0.90)
            test_apriori_p90 = test_apriori_hints.quantile(0.90)
            model_perf['train_model_p90'].append(train_model_p90.to('cpu'))
            model_perf['train_opt_p90'].append(train_opt_p90.to('cpu'))
            model_perf['train_benchmark_p90'].append(train_benchmark_p90.to('cpu'))
            model_perf['train_apriori_p90'].append(train_apriori_p90.to('cpu'))
            model_perf['test_model_p90'].append(test_model_p90.to('cpu'))
            model_perf['test_opt_p90'].append(test_opt_p90.to('cpu'))
            model_perf['test_benchmark_p90'].append(test_benchmark_p90.to('cpu'))
            model_perf['test_apriori_p90'].append(test_apriori_p90.to('cpu'))

            train_model_median = train_model_hints.median()
            train_opt_median = opt_l_train.mean(dim=1).median()
            train_benchmark_median = hint_l_train[:,0].median()
            train_apriori_median = train_apriori_hints.median()
            test_model_median = test_model_hints.median()
            test_opt_median = opt_l_test.mean(dim=1).median()
            test_benchmark_median = hint_l_test[:,0].median()
            test_apriori_median = test_apriori_hints.median()
            model_perf['train_model_median'].append(train_model_median.to('cpu'))
            model_perf['train_opt_median'].append(train_opt_median.to('cpu'))
            model_perf['train_benchmark_median'].append(train_benchmark_median.to('cpu'))
            model_perf['train_apriori_median'].append(train_apriori_median.to('cpu'))
            model_perf['test_model_median'].append(test_model_median.to('cpu'))
            model_perf['test_opt_median'].append(test_opt_median.to('cpu'))
            model_perf['test_benchmark_median'].append(test_benchmark_median.to('cpu'))
            model_perf['test_apriori_median'].append(test_apriori_median.to('cpu'))
        else:
            train_model_workload = hint_l_train[:, probs_train.argmax(dim=1)].mean(dim=1).sum()
            train_opt_workload = opt_l_train.mean(dim=1).sum()
            train_benchmark_workload = hint_l_train[:,0].sum()
            test_model_workload = hint_l_test[:, probs_test.argmax(dim=1)].mean(dim=1).sum()
            test_opt_workload = opt_l_test.mean(dim=1).sum()
            test_benchmark_workload = hint_l_test[:,0].sum()
            model_perf['train_model_workload'].append(train_model_workload.to('cpu'))
            model_perf['train_opt_workload'].append(train_opt_workload.to('cpu'))
            model_perf['train_benchmark_workload'].append(train_benchmark_workload.to('cpu'))
            model_perf['test_model_workload'].append(test_model_workload.to('cpu'))
            model_perf['test_opt_workload'].append(test_opt_workload.to('cpu'))
            model_perf['test_benchmark_workload'].append(test_benchmark_workload.to('cpu'))

            train_model_p90 = hint_l_train[:, probs_train.argmax(dim=1)].mean(dim=1).quantile(0.90)
            train_opt_p90 = opt_l_train.mean(dim=1).quantile(0.90)
            train_benchmark_p90 = hint_l_train[:,0].quantile(0.90)
            test_model_p90 = hint_l_test[:, probs_train.argmax(dim=1)].mean(dim=1).quantile(0.90)
            test_opt_p90 = opt_l_test.mean(dim=1).quantile(0.90)
            test_benchmark_p90 = hint_l_test[:,0].quantile(0.90)
            model_perf['train_model_p90'].append(train_model_p90.to('cpu'))
            model_perf['train_opt_p90'].append(train_opt_p90.to('cpu'))
            model_perf['train_benchmark_p90'].append(train_benchmark_p90.to('cpu'))
            model_perf['test_model_p90'].append(test_model_p90.to('cpu'))
            model_perf['test_opt_p90'].append(test_opt_p90.to('cpu'))
            model_perf['test_benchmark_p90'].append(test_benchmark_p90.to('cpu'))

            train_model_median = hint_l_train[:, probs_train.argmax(dim=1)].mean(dim=1).median()
            train_opt_median = opt_l_train.mean(dim=1).median()
            train_benchmark_median = hint_l_train[:,0].median()
            test_model_median = hint_l_test[:, probs_train.argmax(dim=1)].mean(dim=1).median()
            test_opt_median = opt_l_test.mean(dim=1).median()
            test_benchmark_median = hint_l_test[:,0].median()
            model_perf['train_model_median'].append(train_model_median.to('cpu'))
            model_perf['train_opt_median'].append(train_opt_median.to('cpu'))
            model_perf['train_benchmark_median'].append(train_benchmark_median.to('cpu'))
            model_perf['test_model_median'].append(test_model_median.to('cpu'))
            model_perf['test_opt_median'].append(test_opt_median.to('cpu'))
            model_perf['test_benchmark_median'].append(test_benchmark_median.to('cpu'))

        model_perf['train_loss'].append(model_loss)

        if device == 'mps':
            del model, optimizer
            del probs_train, probs_test
            del X_train, hint_l_train, opt_l_train, targets_l_train
            del X_test, hint_l_test, opt_l_test, targets_l_test
            gc.collect()
            torch.mps.empty_cache()

    print(f'[{datetime.now().isoformat()}] {type(neural_model).__name__}{f"" if exp is None else " [Experiment #" + str(exp) + "]"} - Training on {device.upper()} Complete')
    results = {
        'model_name': type(neural_model).__name__,
        'loss_name': type(loss_model).__name__,
        'post_process': post_process,
        'PCs': components,
        'epochs': epochs,
        'k': k,
        'device': device,
        **model_perf,
    }
    return results


def generate_experiments(binary_model: bool, run_mode: str, parallel: bool, X: torch.Tensor, hint_l: torch.Tensor, opt_l: torch.Tensor, targets_l: torch.Tensor, k: int = 5) -> list[dict]:
    """
    helper function to create the parameter grid for each experiment / model training session

    Args:
        binary_model (bool): indicator for training binary classification or multiclass classification task
        run_mode (str): determines the set of models for which experiments are ran
        parallel (bool): which devices to split expirements across
        X (torch.Tensor): input data (m x p)
        hint_l (torch.Tensor): hint latencies (m x 49)
        opt_l (torch.Tensor): optimal hint latencies (m x 49)
        targets_l (torch.Tensor): optimal hints binary encoded (m x 49)
        k (int, optional): the number of cross-validation folds to perform. Defaults to 5.
        
    Returns:
        list[dict]: a set of dictionaries containing the parameters for each experiment
    """    
    epochs = {
        '2H': 500,
        '3H': 1000,
        '4H': 1000,
        '5H': 1000,
        '6H': 1000,
        '7H': 1000,
    }
    expirements = list()
    pcs = [5, 50, 120]
    post_proc = [True, False]
    models = BINARY_RUN_MODES[run_mode] if binary_model else MULTICLASS_RUN_MODES[run_mode] 
    for m in models:
        params_grid = dict(
            X=X,
            hint_latencies=hint_l,
            opt_latencies=opt_l,
            targets=targets_l, 
            k=k,
        )

        for pc in pcs:
            neural_model = m(pc)
            loss_model = nn.BCEWithLogitsLoss() if binary_model else ERLoss()
            
            params_grid['neural_model'] = neural_model
            params_grid['loss_model'] = loss_model
            params_grid['components'] = pc
            params_grid['epochs'] = epochs.get(type(neural_model).__name__.split('_')[1], 1000)

            if parallel:
                params_grid['device'] = 'mps' if type(neural_model).__name__.split('_')[1] in ['5H', '6H', '7H'] else 'cpu'
            else:
                params_grid['device'] = 'cpu'

            for proc in post_proc:
                params_grid['post_process'] = proc
                expirements.append(deepcopy(params_grid))
    return expirements


def load_data() -> pd.DataFrame:
    """
    helper function to load and combine datasets

    Returns:
        pd.DataFrame: a dataset comprised of JOB queries
    """    
    job_df = pd.read_csv(
        './data/job.csv',
        converters={
            'hint_list': eval,
            'runtime_list': eval,
        }
    )
    ceb_df = pd.read_csv(
        './data/ceb.csv',
        converters={
            'hint_list': eval,
            'runtime_list': eval,
        }
    )
    data = pd.concat([job_df, ceb_df]).reset_index(drop=True)
    
    data["mean_runtime"] = data.runtime_list.apply(lambda x: np.mean(x)) # compute mean runtime for each query plan
    data["sd_runtime"] = data.runtime_list.apply(lambda x: np.std(x)) # compute sd runtime for each query plan
    data["sql"] = data.sql.apply(lambda x: x.strip('\n'))
    return data


def setup_parser():
    parser = argparse.ArgumentParser(
        prog='TrainModels',
        description='Main driver module for training deep learning models on the task of database hint steering.'
    )
    parser.add_argument(
        '--binary',
        action='store_true',
        help='Indicator for training binary classification or multiclass classification task.',
        required=False,
    )
    parser.add_argument(
        '--run-mode',
        action='store',
        choices=['EROnly', 'BCEOnly', 'LNOnly', 'NLNOnly', 'ALL',],
        help='The run mode determines the set of models that are evaluated.',
        required=True,
    )
    parser.add_argument(
        '--processes',
        action='store',
        help=
            'The number of CPU cores to use during training. '
            'Default will use N-1 available cores. '
            'Argument above N or equal to 0 will use N-1 CPU cores. '
            'Argument of -1 or will use all available cores.',
        required=False,
        type=int,
        default=cpu_count()-1,
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='To employ multiple CPU cores for parallel training.',
        required=False,
    )
    return parser