import json, os, re, ast
from difflib import SequenceMatcher
from AutoFL import name_utils
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import from_networkx
from collections import defaultdict

D4J_BUG_INFO_DIR = './AutoFL/data/defects4j'
BIP_BUG_INFO_DIR = './AutoFL/data/bugsinpy'

class D4JProcessing():
    def __init__(self, bug_name) -> None:
        self._method_lists = self._load_method_lists(bug_name)
        self._test_lists = self._load_test_lists(bug_name)
        self._field_lists = self._load_field_lists(bug_name)
        self._test_signatures = [test['signature'] for test in self._test_lists]
        self._field_signatures = [field['signature'] for field in self._field_lists] 
        self._method_signatures = [method['signature'] for method in self._method_lists]
    
    def process_get_failing_tests_covered_methods_for_class(self, class_name):

        for method in self._method_lists:
            if method["class_name"] == class_name:
                return class_name
            elif class_name in self._test_signatures:
                return class_name
            else:
                return None

    def process_get_code_snippet(self,signature):
        if signature in self._field_signatures:
            return signature

        method, candidates = self.get_matching_method_or_candidates(signature, 5)
        if method:
            return method['signature']
        
        if len(candidates) == 0 and not name_utils.is_method_signature(signature):
            candidates = [field for field in self._field_lists if name_utils.get_base_name(signature) in field["signature"]][:5]

        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]['signature']
        else:
            return None

    def process_get_comments(self, signature):
        if signature in self._field_signatures:
            return signature

        method, candidates = self.get_matching_method_or_candidates(signature, 5)
        if method:
            return method['signature']

        if len(candidates) == 0 and not name_utils.is_method_signature(signature):
            candidates = [field for field in self._field_lists if name_utils.get_base_name(signature) in field["signature"]][:5]

        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]['signature']
        else:
            return None

    def process_answer(self, answer):
        pred_exprs = answer.splitlines()
        matching_methods_signatures = {
            pred_expr: self.get_matching_method_signatures(pred_expr)
            for pred_expr in pred_exprs
        }

        return matching_methods_signatures

    def get_matching_method_or_candidates(self, pred_expr: str, num_max_candidates:int=None) -> tuple:
        candidates = {}

        short_method_name = name_utils.get_method_name(pred_expr)

        search_lists = []
        search_lists += self._method_lists
        search_lists += self._test_lists

        for method in search_lists:
            if name_utils.lenient_matcher(pred_expr, method['signature']):
                return (method, None)
            if short_method_name in method["signature"]:
                candidates[method["signature"]] = method

        if len(candidates) == 0:
            return None, []

        priority, candidate_signatures = self.get_highest_priority_candidates(
             pred_expr, list(candidates.keys()), num_max_candidates=num_max_candidates)

        assert (num_max_candidates is None or
                len(candidate_signatures) <= num_max_candidates)

        if priority == 0 and len(candidate_signatures) == 1:
            return (candidates[candidate_signatures[0]], None)
        else:
            return (None, [candidates[sig] for sig in candidate_signatures])

    def get_matching_method_signatures(self, pred_expr):
        return [
            signature for signature in self._method_signatures
            if name_utils.lenient_matcher(pred_expr, signature)
        ]
    
    def get_highest_priority_candidates(self, pred_expr: str, candidates: list,
                                        num_max_candidates:int=None):
        def _compute_similarity(method_name_1, arg_types_1, method_name_2, arg_types_2):
            # (method name similarity , short name matching, arg type similarity)
            return (
                SequenceMatcher(None, method_name_1, method_name_2).ratio(),
                method_name_1[-1] == method_name_2[-1],
                SequenceMatcher(None, arg_types_1, arg_types_2).ratio()
            )

        def _get_priority(method_similarity: float, short_name_match: bool,
                          arg_type_similarity: float):
            if method_similarity == 1.0:
                assert short_name_match
                priority = 0 if arg_type_similarity == 1.0 else 1
            else:
                priority = 2 if short_name_match else 3
            return priority

        assert len(candidates) > 0

        pred_method_name, pred_arg_types = name_utils.get_method_name_and_argument_types(pred_expr)
        similarities = defaultdict(list)
        for candidate in candidates:
            cand_method_name, cand_arg_types = name_utils.get_method_name_and_argument_types(candidate)
            similarity = _compute_similarity(pred_method_name, pred_arg_types,
                                            cand_method_name, cand_arg_types)
            priority = _get_priority(*similarity)
            similarities[priority].append((similarity, candidate))
        assert sum(len(v) for v in similarities.values()) == len(candidates)
        assert len(similarities) > 0

        highest_priority = min(similarities.keys())
        candidates = list(map(lambda t: t[1],
                              sorted(similarities[highest_priority], key=lambda t: t[0], reverse=True)))
        if num_max_candidates is not None:
            candidates = candidates[:num_max_candidates]
        return highest_priority, candidates
    
    def _load_method_lists(self, bug_name):
        with open(os.path.join(D4J_BUG_INFO_DIR, bug_name, "snippet.json")) as f:
            method_list = json.load(f)
        return method_list
    
    def _load_test_lists(self, bug_name):
        with open(os.path.join(D4J_BUG_INFO_DIR, bug_name, "test_snippet.json")) as f:
            test_list = json.load(f)
        return test_list
    
    def _load_field_lists(self, bug_name):
        with open(os.path.join(D4J_BUG_INFO_DIR, bug_name, "field_snippet.json")) as f:
            field_list = json.load(f)
        return field_list

class BIPProcessing():
    def __init__(self, bug_name) -> None:
        self._method_lists = self._load_method_lists(bug_name)
        self._test_lists = self._load_test_lists(bug_name)
        self._field_lists = self._load_field_lists(bug_name)
        self._test_signatures = [test['signature'] for test in self._test_lists]
        self._field_signatures = [field['signature'] for field in self._field_lists] 
        self._method_signatures = [method['signature'] for method in self._method_lists]
    
    def process_get_failing_tests_covered_classes(self, package_name):

        return package_name
    
    def process_get_failing_tests_covered_methods_for_class(self, class_name):

        for method in self._method_lists:
            if method["class_name"] == class_name:
                return class_name
            elif class_name in self._test_signatures:
                return class_name
            else:
                return None
            

    def process_get_code_snippet(self,signature):
        if signature in self._field_signatures:
            return signature

        method, candidates = self.get_matching_method_or_candidates(signature, 5)
        if method:
            return method['signature']

        if len(candidates) == 0 and name_utils.is_method_signature(signature):
            candidates = [field for field in self._field_lists if name_utils.get_base_name(signature) in field["signatures"]][:5]

        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]['signature']
        else:
            return None

    def process_answer(self, answer):
        pred_exprs = answer.splitlines()
        matching_methods_signatures = {
            pred_expr: self.get_matching_method_signatures(pred_expr)
            for pred_expr in pred_exprs
        }

        return matching_methods_signatures

    def get_matching_method_or_candidates(self, pred_expr: str, num_max_candidates:int=None) -> tuple:
        candidates = {}

        short_method_name = name_utils.get_method_name(pred_expr)

        search_lists = []
        search_lists += self._method_lists
        search_lists += self._test_lists

        for method in search_lists:
            if name_utils.python_lenient_matcher(pred_expr, method['signature']):
                return (method, None)
            if short_method_name in method["signature"]:
                candidates[method["signature"]] = method

        if len(candidates) == 0:
            return None, []

        priority, candidate_signatures = self.get_highest_priority_candidates(
             pred_expr, list(candidates.keys()), num_max_candidates=num_max_candidates)

        assert (num_max_candidates is None or
                len(candidate_signatures) <= num_max_candidates)

        if priority == 0 and len(candidate_signatures) == 1:
            # exact match
            return (candidates[candidate_signatures[0]], None)
        else:
            return (None, [candidates[sig] for sig in candidate_signatures])

    def get_matching_method_signatures(self, pred_expr):
        return [
            signature for signature in self._method_signatures
            if name_utils.python_lenient_matcher(pred_expr, signature)
        ]
    
    def get_highest_priority_candidates(self, pred_expr: str, candidates: list,
                                        num_max_candidates:int=None):
        def _compute_similarity(method_name_1, arg_types_1, method_name_2, arg_types_2):
            return (
                SequenceMatcher(None, method_name_1, method_name_2).ratio(),
                method_name_1[-1] == method_name_2[-1],
                SequenceMatcher(None, arg_types_1, arg_types_2).ratio()
            )

        def _get_priority(method_similarity: float, short_name_match: bool,
                          arg_type_similarity: float):
            if method_similarity == 1.0:
                assert short_name_match
                priority = 0 if arg_type_similarity == 1.0 else 1
            else:
                priority = 2 if short_name_match else 3
            return priority

        assert len(candidates) > 0

        pred_method_name, pred_arg_types = name_utils.get_method_name_and_argument_types(pred_expr)
        similarities = defaultdict(list)
        for candidate in candidates:
            cand_method_name, cand_arg_types = name_utils.get_method_name_and_argument_types(candidate)
            similarity = _compute_similarity(pred_method_name, pred_arg_types,
                                            cand_method_name, cand_arg_types)
            priority = _get_priority(*similarity)
            similarities[priority].append((similarity, candidate))
        assert sum(len(v) for v in similarities.values()) == len(candidates)
        assert len(similarities) > 0

        highest_priority = min(similarities.keys())
        candidates = list(map(lambda t: t[1],
                              sorted(similarities[highest_priority], key=lambda t: t[0], reverse=True)))
        if num_max_candidates is not None:
            candidates = candidates[:num_max_candidates]
        return highest_priority, candidates
    
    def _load_method_lists(self, bug_name):
        with open(os.path.join(BIP_BUG_INFO_DIR, bug_name, "snippet.json")) as f:
            method_list = json.load(f)
        return method_list
    
    def _load_test_lists(self, bug_name):
        with open(os.path.join(BIP_BUG_INFO_DIR, bug_name, "test_snippet.json")) as f:
            test_list = json.load(f)
        return test_list
    
    def _load_field_lists(self, bug_name):
        with open(os.path.join(BIP_BUG_INFO_DIR, bug_name, "field_snippet.json")) as f:
            field_list = json.load(f)
        return field_list

def d4j_get_reasoning_paths_and_args(bug_name):
    arg_set = set()
    reasoning_paths = []
    dp = D4JProcessing(bug_name)

    for i in range(1, 11):
        result_file = f"./AutoFL/results/d4j_autofl_{i}/gpt-4o/XFL-{bug_name}.json"
        with open(result_file, 'r') as f:
            content = json.load(f)
            
        function_calls = []
        dialog = content["messages"]
        for j, m in enumerate(dialog):
            if m.get("function_call"):
                function_name = m["function_call"]["name"]
                function_args = json.loads(m["function_call"]["arguments"])
                

                if function_name == "get_failing_tests_covered_classes":
                    # print(**function_args)
                    reformated_arg = None
                elif function_name == "get_failing_tests_covered_methods_for_class":
                    reformated_arg = dp.process_get_failing_tests_covered_methods_for_class(**function_args)
                elif function_name == "get_code_snippet":
                    reformated_arg = dp.process_get_code_snippet(**function_args)
                elif function_name == "get_comments":
                    reformated_arg = dp.process_get_comments(**function_args)

                if reformated_arg:
                    arg_set.add(reformated_arg)
                processed_function_call = {"name": function_name, "arguments": reformated_arg}
                function_calls.append(processed_function_call)

        answer_signatures_dict = dp.process_answer(dialog[-1]["content"])
        for answer, signatures in answer_signatures_dict.items():
            for sig in signatures:
                arg_set.add(sig)

        reasoning_paths.append({"function_calls": function_calls, "answer": answer_signatures_dict})
    return reasoning_paths, arg_set

def bip_get_reasoning_paths_and_args(bug_name):
    arg_set = set()
    reasoning_paths = []
    bp = BIPProcessing(bug_name)

    for i in range(1, 11):
        result_file = f"./AutoFL/results/bip_autofl_{i}/gpt-4o/XFL-{bug_name}.json"
        with open(result_file, 'r') as f:
            content = json.load(f)
            
        function_calls = []
        dialog = content["messages"]
        for j, m in enumerate(dialog):
            if m.get("function_call"):
                function_name = m["function_call"]["name"]
                function_args = json.loads(m["function_call"]["arguments"])
                
                if function_name == "get_covered_packages":
                    reformated_arg = None
                elif function_name == "get_failing_tests_covered_classes":
                    reformated_arg = bp.process_get_failing_tests_covered_classes(**function_args)
                elif function_name == "get_failing_tests_covered_methods_for_class":
                    reformated_arg = bp.process_get_failing_tests_covered_methods_for_class(**function_args)
                elif function_name == "get_code_snippet":
                    reformated_arg = bp.process_get_code_snippet(**function_args)
                else:
                    print(function_name)

                if reformated_arg:
                    arg_set.add(reformated_arg)
                processed_function_call = {"name": function_name, "arguments": reformated_arg}
                function_calls.append(processed_function_call)

        answer_signatures_dict = bp.process_answer(dialog[-1]["content"])
        for answer, signatures in answer_signatures_dict.items():
            for sig in signatures:
                arg_set.add(sig)

        reasoning_paths.append({"function_calls": function_calls, "answer": answer_signatures_dict})
    return reasoning_paths, arg_set

def generate_LIM(reasoning_paths_dict, labels_dict, args_dict):
    dataset_F = []
    dataset_FA = []
    dataset_FAA = []
    y = []

    for bug_name in reasoning_paths_dict.keys():
        F_paths = []
        FA_paths = []
        FAA_paths = []
        reasoning_paths = reasoning_paths_dict[bug_name]
        arg_list = list(args_dict[bug_name])

        for rp in reasoning_paths:
            F_path = []
            FA_path = []
            FAA_path = []
            function_calls, answer = rp["function_calls"], rp["answer"]

            for fc in function_calls:
                if fc["name"] == "get_covered_packages":
                    func_vector = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)
                elif fc["name"] == "get_failing_tests_covered_classes":
                    func_vector = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float)
                elif fc["name"] == "get_failing_tests_covered_methods_for_class":
                    func_vector = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float)
                elif fc["name"] == "get_code_snippet":
                    func_vector = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float)
                elif fc["name"] == "get_comments":
                    func_vector = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)
                
                arg = fc["arguments"]
                
                arg_vector = torch.zeros(28, dtype=torch.float)

                if arg:
                    arg_index = arg_list.index(arg)
                    arg_vector[arg_index] = 1
                elif fc["name"] == "get_covered_packages" or (fc["name"] == "get_failing_tests_covered_classes" and bug_name in d4j_bugs):
                    pass
                else:
                    arg_vector[-1] = 1
                func_arg_vector = torch.cat((func_vector, arg_vector))


                F_path.append(func_vector)
                FA_path.append(func_arg_vector)
                FAA_path.append(func_arg_vector)

            
            while len(F_path) < 10:
                F_path.append(torch.zeros(5, dtype=torch.float)) 
                FA_path.append(torch.zeros(33, dtype=torch.float))
                FAA_path.append(torch.zeros(33, dtype=torch.float))
            
            answer_vector = torch.zeros(28, dtype=torch.float)
            for answers in answer.values():
                for a in answers:
                    answer_index = arg_list.index(a)
                    answer_vector[answer_index] = 1
            
            func_answer_vector = torch.cat((torch.zeros(5, dtype=torch.float), answer_vector))
            
            FAA_path.append(func_answer_vector)

            F_paths.append(torch.stack(F_path))
            FA_paths.append(torch.stack(FA_path))
            FAA_paths.append(torch.stack(FAA_path))


        
        dataset_F.append(F_paths)
        dataset_FA.append(FA_paths)
        dataset_FAA.append(FAA_paths)
        y.append(labels_dict[bug_name])

    dataset_F = torch.stack([torch.stack(path) for path in dataset_F])
    dataset_FA = torch.stack([torch.stack(path) for path in dataset_FA])
    dataset_FAA = torch.stack([torch.stack(path) for path in dataset_FAA]) 
    y = torch.tensor(y, dtype=torch.float)

    return dataset_F, dataset_FA, dataset_FAA, y


def generate_LIG(reasoning_paths_dict, labels_dict, args_dict):
    def add_weighted_edge(G, u, v, weight = 1):
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
        else:
            G.add_edge(u, v, weight = weight)
        
    dataset_S = []
    dataset_F = []
    dataset_FA = []
    dataset_FAA = []

    for bug_name in reasoning_paths_dict.keys():
        print(bug_name)
        reasoning_paths = reasoning_paths_dict[bug_name]
        arg_list = list(args_dict[bug_name])

        LIG = nx.DiGraph()
        for _, rp in enumerate(reasoning_paths):
            function_calls, answer = rp["function_calls"], rp["answer"]
            if len(function_calls) == 0:
                continue
            if not LIG.has_node(str(function_calls[0])):
                LIG.add_node(str(function_calls[0]))
            for i, fc in enumerate(function_calls[1:]):
                if not LIG.has_node(str(fc)):
                    LIG.add_node(str(fc))
                add_weighted_edge(LIG, str(function_calls[i]), str(fc))
            
            for answers in answer.values():
                for a in answers:
                    if not LIG.has_node(a):
                        LIG.add_node(a)
                    add_weighted_edge(LIG, str(function_calls[-1]), a)

        S_data = from_networkx(LIG)
        F_data = from_networkx(LIG)
        FA_data = from_networkx(LIG)
        FAA_data = from_networkx(LIG)

        S_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)
        F_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)
        FA_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)
        FAA_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)

        S_nodes_x = []
        F_nodes_x = []
        FA_nodes_x = []
        FAA_nodes_x = []
        for node in LIG.nodes():
            # Function call node
            if LIG.out_degree(node) != 0 or (LIG.out_degree(node) == 0 and node not in arg_list):
                node = ast.literal_eval(node)
                if node["name"] == "get_covered_packages":
                    func_vector = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)
                elif node["name"] == "get_failing_tests_covered_classes":
                    func_vector = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float)
                elif node["name"] == "get_failing_tests_covered_methods_for_class":
                    func_vector = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float)
                elif node["name"] == "get_code_snippet":
                    func_vector = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float)
                elif node["name"] == "get_comments":
                    func_vector = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)
            
                arg = node["arguments"]

                arg_vector = torch.zeros(28, dtype=torch.float)

                if arg:
                    arg_index = arg_list.index(arg)
                    arg_vector[arg_index] = 1
                elif node["name"] == "get_covered_packages" or (node["name"] == "get_failing_tests_covered_classes" and bug_name in d4j_bugs):
                    pass
                else:
                    arg_vector[-1] = 1
                func_arg_vector = torch.cat((func_vector, arg_vector))


                F_nodes_x.append(func_vector)
                FA_nodes_x.append(func_arg_vector)
                FAA_nodes_x.append(func_arg_vector)

            # Answer node
            else:
                func_vector = torch.zeros(5, dtype=torch.float)
                
                answer_vector = torch.zeros(28, dtype=torch.float)
                answer_index = arg_list.index(node)
                answer_vector[answer_index] = 1

                func_answer_vector = torch.cat((torch.zeros(5, dtype=torch.float), answer_vector))

                F_nodes_x.append(func_vector)
                FA_nodes_x.append(func_arg_vector)
                FAA_nodes_x.append(func_answer_vector)
            
            landscape_vector = torch.ones(5, dtype=torch.float)

            S_nodes_x.append(landscape_vector)

        S_x_stack = np.vstack(S_nodes_x)
        F_x_stack = np.vstack(F_nodes_x)
        FA_x_stack = np.vstack(FA_nodes_x)
        FAA_x_stack = np.vstack(FAA_nodes_x)

        S_data.x = torch.tensor(S_x_stack, dtype=torch.float)
        F_data.x = torch.tensor(F_x_stack, dtype=torch.float)
        FA_data.x = torch.tensor(FA_x_stack, dtype=torch.float)
        FAA_data.x = torch.tensor(FAA_x_stack, dtype=torch.float)

        S_data.y = torch.tensor([labels_dict[bug_name]], dtype=torch.float)
        F_data.y = torch.tensor([labels_dict[bug_name]], dtype=torch.float)
        FA_data.y = torch.tensor([labels_dict[bug_name]], dtype=torch.float)
        FAA_data.y = torch.tensor([labels_dict[bug_name]], dtype=torch.float)

        dataset_S.append(S_data)
        dataset_F.append(F_data)
        dataset_FA.append(FA_data)
        dataset_FAA.append(FAA_data)
    return dataset_S, dataset_F, dataset_FA, dataset_FAA


if __name__ == '__main__':
    d4j_bugs = os.listdir('./AutoFL/data/defects4j')
    d4j_combined_results_file = './AutoFL/combined_fl_results/d4j_gpt4o_results_R10_full.json'
    with open(d4j_combined_results_file, 'r') as f:
        d4j_combined_result = json.load(f)

    bip_bugs = os.listdir('./AutoFL/data/bugsinpy')
    bip_combined_results_file = './AutoFL/combined_fl_results/bip_gpt4o_results_R10_full.json'
    with open(bip_combined_results_file, 'r') as f:
        bip_combined_result = json.load(f)

    # No interaction with the LLM has occurred
    not_work = ["scrapy_20", "keras_9", "keras_14", "keras_45", "tornado_3", "tornado_11"]

    print("------------Generating Reasoning Paths Dataset------------")
    reasoning_paths_dict = dict()
    labels_dict = dict()
    args_dict = dict()
    print("For Defects4J")
    for bug_name in tqdm(d4j_bugs):
        if bug_name in not_work:
            continue
        buggy_methods = d4j_combined_result["buggy_methods"][bug_name]
        if len(buggy_methods) == 1:
            reasoning_paths, arg_set = d4j_get_reasoning_paths_and_args(bug_name)
            reasoning_paths_dict[bug_name] = reasoning_paths
            args_dict[bug_name] = arg_set

            method_name, method_info = next(iter(buggy_methods.items()))
            if method_info.get("autofl_rank") == 1:
                labels_dict[bug_name] = 1
            else:
                labels_dict[bug_name] = 0

    d4j_num = len(args_dict)

    print("For BugsInPy")
    for bug_name in tqdm(bip_bugs):
        if bug_name in not_work:
            continue
        buggy_methods = bip_combined_result["buggy_methods"][bug_name]
        if len(buggy_methods) == 1:
            reasoning_paths, arg_set = bip_get_reasoning_paths_and_args(bug_name)
            
            reasoning_paths_dict[bug_name] = reasoning_paths
            args_dict[bug_name] = arg_set

            method_name, method_info = next(iter(buggy_methods.items()))
            if method_info.get("autofl_rank") == 1:
                labels_dict[bug_name] = 1
            else:
                labels_dict[bug_name] = 0

    bip_num = len(args_dict) - d4j_num
    print(f"From d4j: {d4j_num}")
    print(f"From bip: {bip_num}")
    print(f"Total: {len(args_dict)}")
    print('------------------Successfully Generated----------------')

    lstm_F, lstm_FA, lstm_FAA, lstm_y = generate_LIM(reasoning_paths_dict, labels_dict, args_dict)

    gcn_S, gcn_F, gcn_FA, gcn_FAA = generate_LIG(reasoning_paths_dict, labels_dict, args_dict)

    torch.save({
        "dataset_F": lstm_F,
        "dataset_FA": lstm_FA,
        "dataset_FAA": lstm_FAA,
        "y": lstm_y
    }, "./data/lstm_dataset.pth")
    print("LSTM datasets saved lstm_dataset.pth")

    torch.save({
        "dataset_S": gcn_S,
        "dataset_F": gcn_F,
        "dataset_FA": gcn_FA,
        "dataset_FAA": gcn_FAA,
    }, "data/gcn_dataset.pt")
    print("GCN datasets saved to gcn_dataset.pt")

    all_bugs = list(args_dict.keys())

    with open('./bugs_list.txt', 'w') as f:
        for bug in all_bugs:
            f.write(f"{bug}\n")







