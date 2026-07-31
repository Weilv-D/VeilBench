"""模型 API 封装，支持思考模式"""

import time
from typing import Optional
from openai import OpenAI
from config import MODELS, MAX_TOKENS, TIMEOUT


class ModelClient:
    """
    模型客户端。
    - 被评测模型：传入 model_key，自动读取 MODELS 配置
    - 评委模型：传入 judge_config dict
    """

    def __init__(self, model_key: str, judge_config: dict = None):
        if judge_config:
            self.model_key = model_key
            self.config = judge_config
            self.is_thinking_mode = False
            self.reasoning_effort = None
        else:
            self.model_key = model_key
            self.config = MODELS[model_key]
            self.is_thinking_mode = self.config.get("thinking_mode", False)
            self.reasoning_effort = self.config.get("reasoning_effort")

        self.model_max_tokens = self.config.get("max_tokens", MAX_TOKENS)

        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"],
            timeout=TIMEOUT,
        )
        self.model_id = self.config["model_id"]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> dict:
        """调用模型生成回复，自动处理思考模式参数"""
        if max_tokens is None:
            max_tokens = self.model_max_tokens
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # 思考模式不支持 temperature
        if not self.is_thinking_mode:
            kwargs["temperature"] = temperature if temperature is not None else 0.7

        # DeepSeek 思考模式参数
        if self.is_thinking_mode and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["thinking"] = {"type": "enabled"}

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(**kwargs)
            elapsed = time.time() - start_time

            msg = response.choices[0].message
            content = msg.content or ""
            reasoning_content = getattr(msg, "reasoning_content", None) or None
            if not content and reasoning_content:
                content = reasoning_content
            if not content:
                return {
                    "success": False,
                    "content": "",
                    "reasoning_content": reasoning_content,
                    "elapsed_time": elapsed,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "error": "Empty response from model",
                }
            usage = response.usage

            return {
                "success": True,
                "content": content,
                "reasoning_content": reasoning_content,
                "elapsed_time": elapsed,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "error": None,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "content": "",
                "reasoning_content": None,
                "elapsed_time": elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "error": str(e),
            }

    def __repr__(self):
        return f"ModelClient({self.model_key})"
