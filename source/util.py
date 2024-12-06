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

"""Utilities for training and testing."""

import argparse
from collections.abc import Mapping
import logging
import os
import sys
import time
from typing import Any


class _DebugHook(object):
  instance = None

  def __call__(self, *args, **kwargs):
    if self.instance is None:
      from IPython.core import ultratb  # pylint: disable=g-import-not-at-top
      self.instance = ultratb.FormattedTB(
          mode="Plain", color_scheme="Linux", call_pdb=1)
    return self.instance(*args, **kwargs)


sys.excepthook = _DebugHook()


def parse_args() -> argparse.Namespace:
  """Parse command line arguments."""

  parser = argparse.ArgumentParser("")
  parser.add_argument(
      "-a", "--artifacts", default="artifacts",
      help="folder for all the artifacts", required=False)
  parser.add_argument("-c", "--config",
                      help="yaml configuration file", required=True)
  parser.add_argument("-s", "--split",
                      help="data split to train / test on", default=None)
  parser.add_argument("-n", "--num-iteration",
                      help="number of training iterations",
                      type=int, default=2000)
  parser.add_argument("-o", "--output-dir",
                      help="directory to store logs and checkpoints",
                      default="experiment/")
  return parser.parse_args()


def create_working_directory(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
) -> str:
  """Creates a working directory.

  Args:
    args: args
    cfg: config dict
  Returns:
    working directory
  """
  config = os.path.splitext(os.path.basename(args.config))[0]
  time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
  working_dir = os.path.join(args.output_dir, cfg["dataset"],
                             f"{config}_{args.split}_{time_str}")
  os.makedirs(working_dir)
  return working_dir


def create_logger(working_dir: str) -> logging.Logger:
  """Create a logger with both stream and file handlers.

  Args:
    working_dir: working directory
  Returns:
    logger
  """
  logger = logging.getLogger("")
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler()
  logger.addHandler(handler)
  log_file = os.path.join(working_dir, "log.txt")
  handler = logging.FileHandler(log_file)
  logger.addHandler(handler)
  return logger
