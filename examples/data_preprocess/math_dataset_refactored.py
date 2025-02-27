# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'DigitalLearningGmbH/MATH-lighteval'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    ##### MINWU FIX: For evaluation, we use MATH500 instead to same some inference compute. 
    data_source_500 = 'HuggingFaceH4/MATH-500'
    print(f"Loading the {data_source_500} dataset from huggingface...", flush=True)
    dataset_500 = datasets.load_dataset(data_source_500, trust_remote_code=True)

    train_dataset = dataset['train']
    # test_dataset = dataset['test']
    test_dataset = dataset_500['test']

    ##### MINWU FIX: We follow the deepseek-r1 paper for the prompt. (ref: https://arxiv.org/abs/2501.12948)
    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    instruction_following = """
    Answer the question below in the specified format. 
    First, carefully think through the reasoning process. Then, provide a refined solution explaining the steps of the solution. 
    Enclose the reasoning process within <think> </think> tags and the final solution within <answer> </answer> tags, i.e., <think> reasoning process here </think> <answer> solution here </answer>. 
    Ensure the final answer in the solution is formatted within \\boxed{}.
    """

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = instruction_following + '\n' + 'Question:' + 'question'

            answer = example.pop('solution')
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
