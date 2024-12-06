# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dataset module."""

import ast
from collections.abc import Mapping, Sequence
import csv
import glob
import json
import os
import random
import re
from typing import Any

from nltk import tokenize
import numpy as np


class Arithmetic:
  """Artithmetic dataset."""

  def __init__(
      self,
      path,
      base=10,
      num_train=900,
      num_valid=100,
      num_test=100,
  ):
    """Init arithmetic dataset.

    Args:
      path: path to the dataset
      base: base of the arithmetic
      num_train: number of training samples
      num_valid: number of validation samples
      num_test: number of test samples

    Raises:
      FileNotFoundError: if dataset files are not found
    """

    self.path = path
    self.base = base
    self.num_train = num_train
    self.num_valid = num_valid
    self.num_test = num_test

    train_files = glob.glob(os.path.join(path, f"base-{base}/*_train.txt"))
    test_files = glob.glob(os.path.join(path, f"base-{base}/*_test.txt"))
    if not train_files or not test_files:
      raise FileNotFoundError(f"Can't find dataset files in `{path}`.")

    queries = []
    answers = []
    levels = []
    num_samples = []
    for txt_file in train_files + test_files:
      num_sample = 0
      with open(txt_file, "r") as fin:
        for line in fin:
          query = line.strip().split("+")
          answer = sum(int(x, base) for x in query)
          answer = np.base_repr(answer, base)
          level = f"{len(query[0])} digits"
          queries.append(query)
          answers.append(answer)
          levels.append(level)
          num_sample += 1
      num_samples.append(num_sample)

    total_train = sum(num_samples[:len(train_files)])
    total_test = sum(num_samples[len(train_files):])
    train_indices = random.sample(range(total_train), num_train)
    test_indices = random.sample(range(total_train, total_train + total_test),
                                 num_valid + num_test)
    indices = train_indices + test_indices
    self.queries = [queries[i] for i in indices]
    self.answers = [answers[i] for i in indices]
    self.levels = [levels[i] for i in indices]

  def get_split(
      self,
      split: str = "test",
  ) -> Sequence[Mapping[str, Any]]:
    """Get dataset split.

    Args:
      split: split name
    Returns:
      List of samples
    """
    if split == "train":
      indices = range(self.num_train)
    elif split == "valid":
      indices = range(self.num_train, self.num_train + self.num_valid)
    elif split == "test":
      indices = range(len(self) - self.num_test, len(self))
    else:
      raise ValueError(f"Unknown split `{split}`")
    return [self[i] for i in indices]

  def evaluate(
      self,
      truth: str,
      pred: str,
  ):
    """Evaluate truth and pred."""
    return truth.lower() in tokenize.word_tokenize(pred.lower())

  def __getitem__(
      self,
      index,
  ):
    return {
        "query": self.queries[index],
        "answer": self.answers[index],
        "level": self.levels[index],
        "base": self.base,
    }

  def __len__(self):
    return len(self.queries)


class CLUTRR:
  """CLUTRR dataset."""

  def __init__(
      self,
      path,
      num_train=2000,
      num_valid=200,
      num_test=200,
  ):
    self.path = path
    self.num_train = num_train
    self.num_valid = num_valid
    self.num_test = num_test

    train_files = glob.glob(os.path.join(path, "*_train.csv"))
    test_files = glob.glob(os.path.join(path, "*_test.csv"))
    if not train_files or not test_files:
      raise FileNotFoundError(f"Can't find dataset files in `{path}`.")

    documents = []
    paths = []
    queries = []
    answers = []
    levels = []
    num_samples = []
    for csv_file in train_files + test_files:
      num_sample = 0
      with open(csv_file, "r") as fin:
        reader = csv.reader(fin)
        fields = next(reader)
        for values in reader:
          document = path = query = answer = level = None
          for field, value in zip(fields, values, strict=True):
            if field == "story":
              document = re.sub(r"[\[\]]", "", value)
            elif field == "f_comb":
              path = value.split("-")
            elif field == "query":
              query = ast.literal_eval(value)
            elif field == "target":
              answer = value
            elif field == "task_name":
              level = f"{value.split('.')[1]} hops"
          documents.append(document)
          paths.append(path)
          queries.append(query)
          answers.append(answer)
          levels.append(level)
          num_sample += 1
      num_samples.append(num_sample)

    total_train = sum(num_samples[:len(train_files)])
    total_test = sum(num_samples[len(train_files):])
    train_indices = random.sample(range(total_train), num_train)
    test_indices = random.sample(range(total_train, total_train + total_test),
                                 num_valid + num_test)
    indices = train_indices + test_indices
    self.documents = [documents[i] for i in indices]
    self.paths = [paths[i] for i in indices]
    self.queries = [queries[i] for i in indices]
    self.answers = [answers[i] for i in indices]
    self.levels = [levels[i] for i in indices]
    self.labels = set(answers)

  def get_split(
      self,
      split="test"
  ):
    """Get dataset split."""
    if split == "train":
      indices = range(self.num_train)
    elif split == "valid":
      indices = range(self.num_train, self.num_train + self.num_valid)
    elif split == "test":
      indices = range(len(self) - self.num_test, len(self))
    else:
      raise ValueError(f"Unknown split `{split}`")
    return [self[i] for i in indices]

  def evaluate(
      self,
      truth,
      pred
  ):
    """Evaluate truth and pred."""
    truth = truth.lower()
    words = tokenize.word_tokenize(pred.lower())
    others = self.labels - {truth}
    return truth in words and not any(label in words for label in others)

  def __getitem__(
      self,
      index
  ):
    return {
        "document": self.documents[index],
        "path": self.paths[index],
        "query": self.queries[index],
        "answer": self.answers[index],
        "level": self.levels[index]
    }

  def __len__(self):
    return len(self.queries)


class ListFunctions:
  """List Functions dataset."""

  def __init__(
      self,
      path: str,
      num_train: int = 8,
      num_valid: int = 8,
      num_test: int = 16
  ) -> None:
    self.path = path
    self.num_train = num_train
    self.num_valid = num_valid
    self.num_test = num_test

    num_sample = num_train + num_valid + num_test
    json_files = sorted(glob.glob(os.path.join(path, "c*.json")))
    if not json_files:
      raise FileNotFoundError(f"Can't find dataset files in `{path}`.")

    levels = []
    concepts = []
    queries = []
    answers = []
    for json_file in json_files:
      with open(json_file, "r") as fin:
        obj = json.load(fin)
      query = []
      answer = []
      for example in obj["examples"]:
        query.append(example["input"])
        answer.append(example["target"])
      concept = re.search(r"(c\d+).json", json_file)
      assert concept is not None
      concept = concept.group(1)
      cid = int(concept[1:])
      if cid <= 80:
        level = "P1"
      elif cid <= 100:
        level = "P2"
      else:
        level = "P3"
      indices = random.sample(range(len(query)), num_sample)
      query = [query[i] for i in indices]
      answer = [answer[i] for i in indices]
      levels.append(level)
      concepts.append(concept)
      queries.append(query)
      answers.append(answer)

    self.levels = levels
    self.concepts = concepts
    self.queries = queries
    self.answers = answers

  def get_split(
      self,
      split: str = "test"
  ) -> Sequence[Mapping[str, Any]]:
    """Get dataset split."""
    if split in {"train", "valid"}:
      train_indices = slice(self.num_train)
      test_indices = slice(self.num_train, self.num_train + self.num_valid)
    elif split == "test":
      train_indices = slice(self.num_train + self.num_valid)
      test_indices = slice(self.num_train + self.num_valid, None)
    else:
      raise ValueError(f"Unknown split `{split}`")

    dataset = []
    for sample in self:
      sample["train_queries"] = sample["queries"][train_indices]
      sample["train_answers"] = sample["answers"][train_indices]
      sample["queries"] = sample["queries"][test_indices]
      sample["answers"] = sample["answers"][test_indices]
      answers = [f"{q} -> {a}"
                 for q, a in zip(sample["queries"], sample["answers"])]
      sample["answer"] = "\n".join(answers)
      dataset.append(sample)
    return dataset

  def evaluate(
      self,
      truth: str,
      pred: str
  ) -> bool:
    """Evaluate truth and pred."""
    pattern = r"(\[[A-Z0-9, ]*\]) ?-> ?(\[[A-Z0-9, ]*\])"
    query2truth = dict(re.findall(pattern, truth))
    query2pred = dict(re.findall(pattern, pred))
    num_correct = 0
    for query, truth in query2truth.items():
      if query in query2pred:
        try:
          truth = ast.literal_eval(truth)
          pred = ast.literal_eval(query2pred[query])
          num_correct += int(truth == pred)
        except (ValueError, SyntaxError):
          pass
    return num_correct / len(query2truth)

  def __getitem__(self, index: int) -> Mapping[str, Any]:
    return {
        "queries": self.queries[index],
        "answers": self.answers[index],
        "level": self.levels[index],
        "concept": self.concepts[index],
    }

  def __len__(self) -> int:
    return len(self.queries)
