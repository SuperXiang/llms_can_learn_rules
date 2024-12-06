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

"""Training, i.e., rule learning file."""

import os
import pprint
import random

import datasets
import models
import prompts
import tqdm
import util
import yaml


def main():
  random.seed(0)
  args = util.parse_args()
  args.split = args.split or "train"

  with open(args.config, "r") as fin:
    cfg = yaml.safe_load(fin.read())
  cfg = cfg[args.split]

  working_dir = util.create_working_directory(args, cfg)
  logger = util.create_logger(working_dir)
  logger.warning(pprint.pformat(vars(args)))
  logger.warning(pprint.pformat(cfg))

  if cfg["dataset"] == "clutrr":
    dataset = datasets.CLUTRR(os.path.join(args.artifacts, "dataset/clutrr"))
  elif cfg["dataset"].startswith("base-"):
    base = int(cfg["dataset"][5:])
    dataset = datasets.Arithmetic(
        os.path.join(args.artifacts, "dataset/arithmetic"), base=base)
  elif cfg["dataset"] == "list_functions":
    dataset = datasets.ListFunctions(
        os.path.join(args.artifacts, "dataset/list_functions"))
  else:
    raise ValueError(f"Unknown dataset `{cfg['datasets']}`")

  train_set = dataset.get_split(args.split)
  max_tokens = cfg.get("max_tokens", 2000)
  if cfg["model"].startswith("gpt"):
    model = models.GPT(cfg["model"], max_tokens=max_tokens)
  elif cfg["model"].startswith("gemini"):
    model = models.Gemini(cfg["model"], max_tokens=max_tokens)
  else:
    raise ValueError(f"Unknown model `{cfg['model']}`")
  function = prompts.PromptFunction.from_yaml(
      os.path.join(args.artifacts, cfg["prompt"]))
  library = prompts.RuleLibrary()

  num_epoch = args.num_iteration // len(train_set)
  train_set = train_set * num_epoch + random.sample(
      train_set, args.num_iteration % len(train_set))

  total_cost = 0
  num_iteration = 0
  for sample in tqdm.tqdm(train_set):
    truth = sample["answer"]
    pred, cost, rules = function(model, sample)
    logger.warning("rules:")
    for rule in rules:
      logger.warning(rule)
    acc = dataset.evaluate(truth, pred)

    if "concept" in sample:
      concept = sample["concept"]
      rules = [f"[{concept}] {rule}" for rule in rules]
    library.update(rules, acc)
    total_cost += cost
    logger.warning("truth: %s, pred: %s, accuracy: %s", truth, pred, acc)
    logger.warning("total cost: %s", total_cost)

    num_iteration += 1
    if num_iteration % 100 == 0 or num_iteration == args.num_iteration:
      save_file = os.path.join(working_dir, f"library_{num_iteration}.yaml")
      library.save(save_file)
      logger.warning("Save the rule library to `%s`", save_file)


if __name__ == "__main__":
  main()
