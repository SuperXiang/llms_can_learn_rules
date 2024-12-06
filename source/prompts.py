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

"""Prompt function and rule library."""

from collections import defaultdict  # pylint: disable=g-importing-member
from collections.abc import Mapping, Sequence
import logging
import re
from typing import Any

import jinja2
from nltk import tokenize
import yaml


logger = logging.getLogger(__name__)


class RuleLibrary:
  """Rule library that stores learned rules and their statistics."""

  def __init__(self):
    self.count = defaultdict(int)
    self.score = defaultdict(float)

  def update(
      self,
      rules: Sequence[str],
      acc: float,
  ) -> None:
    """Update the posterior of rule confidence, based on the observed acc.

    Args:
      rules: the list of proposed rules.
      acc: the accuracy of applying these rules.
    """
    for rule in rules:
      self.count[rule] += 1
      self.score[rule] += acc

  def save(self, file_name: str):
    """Save the rule library to a file."""
    data = {}
    for rule in self.count:
      data[rule] = [self.count[rule], self.score[rule]]
    with open(file_name, "w") as fout:
      fout.write(yaml.dump(data, width=1000, default_flow_style=None))

  def load(self, file_name: str):
    """Load the rule library from a file."""
    with open(file_name, "r") as fin:
      data = yaml.safe_load(fin.read())
    for rule in data:
      num_recall, num_correct = data[rule]
      self.count[rule] = num_recall
      self.score[rule] = num_correct

  def to_prompt(
      self,
      min_coverage: int = 2,
      min_confidence: float = 0.,
  ) -> Sequence[str]:
    """Convert the rule library to a list of prompts.

    Args:
      min_coverage: threshold of minimal rule coverage
      min_confidence: threshold of minimal confidence
    Returns:
      List of rules.
    """
    rules = {}
    for rule in self.count:
      coverage = self.count[rule]
      confidence = self.score[rule] / self.count[rule]
      if coverage >= min_coverage and confidence >= min_confidence:
        rules[rule] = confidence
    return rules


class PromptFunction:
  """An LLM-based function defined by a prompt string."""

  def __init__(
      self,
      prompt: str,
      system: str = None,
      pattern: str = None,
      stop: str = None,
      return_last: bool = False,
      **kwargs
  ) -> None:
    self.prompt = jinja2.Template(prompt)
    if system is not None:
      self.system = jinja2.Template(system)
    else:
      self.system = None
    self.pattern = pattern
    self.stop = stop
    self.return_last = return_last
    self.kwargs = kwargs

  @classmethod
  def from_yaml(cls, yaml_file: str, **kwargs) -> "PromptFunction":
    with open(yaml_file, "r") as fin:
      config = yaml.safe_load(fin.read())
      kwargs.update(config)
    return cls(**kwargs)

  def __call__(self, model, sample: Mapping[str, Any]) ->  tuple[
      str, float] | tuple[str, float, Sequence[str]]:
    """Call the model with the sample formatted by prompt.

    Args:
      model: the LLM model.
      sample: the input sample defined by k-v pairs.
    Returns:
      The output of the prompt function and the cost. If pattern is defined,
      additionally return the matches in the output.
    """
    logger.info("<" * 50)
    if self.system is not None:
      system = self.system.render(**sample, **self.kwargs)
      marker = "#" * 20
      logger.info("%s System %s", marker, marker)
      logger.info(system)
      logger.info("%s Prompt %s", marker, marker)
    else:
      system = None
    prompt = self.prompt.render(**sample, **self.kwargs)
    logger.info(prompt)
    logger.info("=" * 50)

    response = model(prompt, system=system, stop=self.stop)
    cost = model.get_cost(prompt, system=system, response=response)
    logger.info(response)
    logger.info(">" * 50)

    sents = []
    for line in re.split(r"\n+", response):
      sents += tokenize.sent_tokenize(line)
    if self.return_last:
      pred = sents[-1]
    else:
      pred = response
    if self.pattern:
      matches = []
      for sent in sents:
        matches += re.findall(self.pattern, sent)
      return pred, cost, matches
    else:
      return pred, cost
