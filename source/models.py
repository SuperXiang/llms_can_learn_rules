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

"""GPT model wrapper."""

from collections.abc import Mapping, Sequence
import os
from typing import Final
import google.generativeai as genai
import openai
# pylint: disable=g-importing-member
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
import tiktoken


class GPT:
  """GPT model object."""

  _MODEL_PRICE: Final[Mapping[str, tuple[float, float]]] = {
      "gpt-3.5-turbo": (0.0005 / 1000, 0.0015 / 1000),
      "gpt-3.5-turbo-0613": (0.0005 / 1000, 0.0015 / 1000),
      "gpt-3.5-turbo-16k-0613": (0.0005 / 1000, 0.0015 / 1000),
      "gpt-4": (0.03 / 1000, 0.06 / 1000),
  }

  def __init__(
      self,
      model: str = "gpt-3.5-turbo",
      temperature: float = 1.0,
      top_p: float = 1.0,
      max_tokens: int = 2000,
  ):
    self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    self.encoding = tiktoken.encoding_for_model(model)
    self.model = model
    self.temperature = temperature
    self.top_p = top_p
    self.max_tokens = max_tokens

  @retry(wait=wait_random_exponential(min=1, max=60),
         stop=stop_after_attempt(10))
  def __call__(
      self,
      prompt: str,
      system: str | None = None,
      stop: str | None = None,
  ):
    messages = []
    if system:
      messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=self.temperature,
        top_p=self.top_p,
        max_tokens=self.max_tokens,
        stop=stop
    )
    return response.choices[0].message.content

  def get_cost(
      self,
      prompt: str,
      response: str | None = None,
      system: str | None = None,
  ) -> float:
    """Get the cost for this request."""
    if system is not None:
      prompt = system + prompt
    num_prompt_token = len(self.encoding.encode(prompt))
    if response:
      num_response_token = len(self.encoding.encode(response))
    else:
      num_response_token = self.max_tokens
    input_price, output_price = self._MODEL_PRICE[self.model]
    return num_prompt_token * input_price + num_response_token * output_price


class Gemini:
  """Wrapper for Gemini models."""

  model_price = {
      "gemini-pro": (0, 0),
  }

  _SETTINGS: Final[Sequence[Mapping[str, str]]] = tuple(
      {
          "category": f"HARM_CATEGORY_{category}",
          "threshold": "BLOCK_NONE",
      } for category in [
          "HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS"]
  )

  def __init__(
      self,
      model: str = "gemini-pro",
      temperature: float = 0.9,
      top_p: float = 1,
      max_tokens: int = 2000,
  ) -> None:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    self.model = model
    self.temperature = temperature
    self.top_p = top_p
    self.max_tokens = max_tokens

  @retry(wait=wait_random_exponential(min=1, max=60),
         stop=stop_after_attempt(10))
  def __call__(
      self,
      prompt: str,
      system: str | None = None,
      stop: str | None = None,
  ):
    config = genai.GenerationConfig(
        temperature=self.temperature,
        top_p=self.top_p,
        max_output_tokens=self.max_tokens,
        stop_sequences=stop
    )
    model = genai.GenerativeModel(self.model, generation_config=config,
                                  safety_settings=self._SETTINGS)
    messages = []
    if system:
      messages.append({"role": "user", "parts": f"{system}\n\n{prompt}"})
    else:
      messages.append({"role": "user", "parts": prompt})
    response = model.generate_content(prompt)
    return response.text

  def get_cost(
      self,
      prompt: str,
      response: str | None = None,
      system: str | None = None,
  ) -> float:
    """Get the cost for this request."""
    if system is not None:
      prompt = system + prompt
    model = genai.GenerativeModel(self.model)
    num_prompt_token = model.count_tokens(prompt).total_tokens
    if response:
      num_response_token = model.count_tokens(response).total_tokens
    else:
      num_response_token = self.max_tokens
    input_price, output_price = self.model_price[self.model]
    return num_prompt_token * input_price + num_response_token * output_price
