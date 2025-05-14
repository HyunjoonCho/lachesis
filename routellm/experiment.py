import argparse
import json
import os
import random

from routers.routers import MatrixFactorizationRouter
from routers.matrix_factorization.train_matrix_factorization import *

import torch
import numpy as np
from sklearn.model_selection import KFold

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

checkpoint_dir = 'routers/matrix_factorization/cv'
probs_dir = 'results/cv'
prompt_path = 'route_data/initial_prompts.json'
embeddings_path = 'route_data/embeddings.json'
combined_fl_dir = '../AutoFL/combined_fl_results/' 

def extract_min_ranks(result, bug_list):
    min_ranks = list() 
    for bug_id in bug_list:
        buggy_methods = result[bug_id]
        min_ranks.append(0 if not buggy_methods else min(map(lambda x: x['autofl_rank'], buggy_methods.values())))
    
    return min_ranks
 
def build_and_store_data(strong_model, strong_result, weak_model, weak_result):
    sorted_bug_list = sorted(strong_result.keys())

    strong_ranks = extract_min_ranks(strong_result, sorted_bug_list)
    weak_ranks = extract_min_ranks(weak_result, sorted_bug_list)
    vs_result = list(map(lambda x: x[0] < x[1], zip(strong_ranks, weak_ranks)))
    vs_data = list()

    for i, _ in enumerate(sorted_bug_list):
        vs_data.append({
            "model_a": strong_model,
            "model_b": weak_model,
            "idx": i,
            "winner": "model_a" if vs_result[i] else "model_b"
        })

    with open(vs_data_path, 'w') as f:
        json.dump(vs_data, f, indent=4)

    with open(embeddings_path) as f:
        embeddings = json.load(f)
    filtered_embeddings = np.array([embeddings[bug_id] for bug_id in sorted_bug_list])
    np.save(filtered_embeddings_path, filtered_embeddings)

def split_and_get_loaders_for(training_data):
    train_size = int(0.75 * len(training_data))       
    train_data = training_data[:train_size]
    val_data = training_data[train_size:]
    
    train_loader = PairwiseDataset(train_data).get_dataloaders(
        batch_size=batch_size, shuffle=True
    )
    val_loader = PairwiseDataset(val_data).get_dataloaders(
        batch_size=8, shuffle=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', '-m1', default='gpt-4o')
    parser.add_argument('--model2', '-m2', default='llama3-8b')
    parser.add_argument('--model1_result', default='d4j_gpt4o_results_R10_full.json')
    parser.add_argument('--model2_result', default='d4j_eol_llama3_R10.json')
    parser.add_argument('--runs', '-r', default=10)
    args = parser.parse_args()

    model1_result_path = os.path.join(combined_fl_dir, args.model1_result)
    model2_result_path = os.path.join(combined_fl_dir, args.model2_result)
 
    with open(model1_result_path) as f:
        data = json.load(f)
        model1_acc1 = data['summary']['acc@1'] / data['summary']['total']
        model1_result = data['buggy_methods']

    with open(model2_result_path) as f:
        data = json.load(f)
        model2_acc1 = data['summary']['acc@1'] / data['summary']['total']
        model2_result = data['buggy_methods']
        
    if model1_acc1 > model2_acc1:
        strong_result, weak_result = model1_result, model2_result
        strong_model, weak_model = args.model1, args.model2
    else:
        weak_result, strong_result = model1_result, model2_result
        weak_model, strong_model = args.model1, args.model2

    os.makedirs('route_data/pairwise', exist_ok=True)
    vs_data_path = f'route_data/pairwise/{strong_model}_vs_{weak_model}_R{args.runs}.json'
    filtered_embeddings_path = f'route_data/pairwise/{strong_model}_{weak_model}_R{args.runs}_embeddings.npy'

    build_and_store_data(strong_model, strong_result, weak_model, weak_result)

    dim = 128
    batch_size = 16
    num_epochs = 30
    alpha = 0.1
    use_proj = True
    lr = 3e-4
    weight_decay = 1e-5
    k = 5

    data = json.load(open(vs_data_path, "r"))

    filtered_data = [
        sample
        for sample in data
        if sample["winner"] in ["model_a", "model_b"]
        and sample["model_a"] != sample["model_b"]
    ]

    with open(prompt_path) as f:
        prompts = json.load(f)

    combined_bug_indices = list(strong_result.keys()) # TODO: weak models may have smaller number of results

    random.shuffle(filtered_data)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(filtered_data)):
        print(f"\nFold {fold + 1}/{k}")

        training_data = [filtered_data[i] for i in train_idx]
        train_loader, val_loader = split_and_get_loaders_for(training_data)
        
        model = MFModel_Train(
            dim=dim,
            num_models=len(MODEL_IDS),
            num_prompts=len(data),
            use_proj=use_proj,
            npy_path=filtered_embeddings_path,
        ).to("cuda")
        
        os.makedirs(f"{checkpoint_dir}/{strong_model}_{weak_model}_R{args.runs}", exist_ok=True) 
        save_path = f"{checkpoint_dir}/{strong_model}_{weak_model}_R{args.runs}/best_{fold + 1}.pt"
        train_loops(
            model,
            train_loader,
            val_loader,
            lr=lr,
            weight_decay=weight_decay,
            alpha=alpha,
            num_epochs=num_epochs,
            device="cuda",
            save_path=save_path,
        )

        router = MatrixFactorizationRouter(
            save_path, 
            strong_model=strong_model, 
            weak_model=weak_model
        )
        
        test_bug_ids = [combined_bug_indices[i] for i in test_idx]
        win_rates = dict()
        for bug_id in test_bug_ids:
            prompt = prompts[bug_id]
            win_rates[bug_id] = router.calculate_strong_win_rate(prompt)

        os.makedirs(f'{probs_dir}/{strong_model}_{weak_model}_R{args.runs}', exist_ok=True)
        with open(f'{probs_dir}/{strong_model}_{weak_model}_R{args.runs}/fold_{fold + 1}.json', 'w') as f:
            json.dump(win_rates, f)
