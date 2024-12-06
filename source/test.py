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

"""Test the llm after learning the rules."""

import collections
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
  args.split = args.split or "test"

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

  test_set = dataset.get_split(args.split)
  max_tokens = cfg.get("max_tokens", 2000)
  if cfg["model"].startswith("gpt"):
    model = models.GPT(cfg["model"], max_tokens=max_tokens)
  elif cfg["model"].startswith("gemini"):
    model = models.Gemini(cfg["model"], max_tokens=max_tokens)
  else:
    raise ValueError(f"Unknown model `{cfg['model']}`")
  if "library" in cfg:
    library = prompts.RuleLibrary()
    library.load(os.path.join(args.artifacts, cfg["library"]))
    rules = library.to_prompt(cfg["min_coverage"], cfg["min_confidence"])
    function = prompts.PromptFunction.from_yaml(
        os.path.join(args.artifacts, cfg["prompt"]), rules=rules)
    logger.warning("Load the rule library from `%s`", cfg["library"])
    logger.warning("min coverage: %d, min confidence: %s, #rules: %d",
                   cfg["min_coverage"], cfg["min_confidence"], len(rules))
  else:
    function = prompts.PromptFunction.from_yaml(
        os.path.join(args.artifacts, cfg["prompt"]))

  level2accs = collections.defaultdict(list)
  total_cost = 0
  for sample in tqdm.tqdm(test_set):
    truth = sample["answer"]
    level = sample["level"]
    pred, cost = function(model, sample)
    acc = dataset.evaluate(truth, pred)

    level2accs[level].append(acc)
    total_cost += cost
    logger.warning("truth: %s, pred: %s, accuracy: %s", truth, pred, acc)
    logger.warning("total cost: %s", total_cost)

  accs = []
  task_accs = []
  for level, level_accs in sorted(level2accs.items()):
    acc = sum(level_accs) / len(level_accs)
    accs.append(acc)
    if isinstance(level_accs[0], float):
      task_acc = sum(x > 0.999 for x in level_accs) / len(level_accs)
      task_accs.append(task_acc)
      logger.warning("[%s] #sample: %d, raw accuracy: %s, task accuracy: %s",
                     level, len(level_accs), acc, task_acc)
    else:
      logger.warning("[%s] #sample: %d, accuracy: %s",
                     level, len(level_accs), acc)
  if task_accs:
    logger.warning("average raw accuracy: %s, average task accuracy: %s",
                   sum(accs) / len(accs), sum(task_accs) / len(task_accs))
  else:
    logger.warning("average accuracy: %s", sum(accs) / len(accs))


if __name__ == "__main__":
  main()
