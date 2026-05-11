from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests


DEFAULT_API_KEY = ""
DEFAULT_API_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = ""
DEFAULT_FALLBACK_MODEL = None
DEFAULT_TIMEOUT = 180
DEFAULT_REQUEST_RETRIES = 2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RETRY_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": message.get("content", ""),
            }
        )
    return normalized


def build_payload(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    enable_thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": _normalize_messages(messages),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if enable_thinking:
        payload["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": True,
            }
        }
    return payload


def extract_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "".join(text_parts)
    return str(content)


def extract_reasoning_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    reasoning_content = message.get("reasoning_content", "")
    if reasoning_content is None:
        return ""
    if isinstance(reasoning_content, str):
        return reasoning_content
    return str(reasoning_content)


def extract_finish_reason(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    finish_reason = choices[0].get("finish_reason", "")
    return "" if finish_reason is None else str(finish_reason)


def _needs_retry_with_more_tokens(response_json: Dict[str, Any]) -> bool:
    return (
        extract_text(response_json).strip() == ""
        and extract_reasoning_text(response_json).strip() != ""
        and extract_finish_reason(response_json) == "length"
    )


def _request_chat_completion(
    *,
    messages: List[Dict[str, Any]],
    api_key: str,
    api_url: str,
    model: str,
    enable_thinking: bool,
    timeout: int,
    request_retries: int,
    max_tokens: int,
    temperature: float,
) -> requests.Response:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = build_payload(
        messages=messages,
        model=model,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    retries = max(0, int(request_retries))
    max_attempts = retries + 1
    last_timeout_exc: Optional[requests.Timeout] = None
    for attempt_index in range(max_attempts):
        try:
            return requests.post(
                f"{api_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.Timeout as exc:
            last_timeout_exc = exc
            if attempt_index + 1 >= max_attempts:
                raise
            # Small linear backoff to avoid hammering the endpoint when it is slow.
            time.sleep(float(attempt_index + 1))
    if last_timeout_exc is not None:
        raise last_timeout_exc
    raise RuntimeError("unreachable")


def _is_no_worker_error(response: requests.Response) -> bool:
    if response.status_code != 503:
        return False
    body = response.text.lower()
    return "no available workers" in body or "all circuits open or unhealthy" in body


def list_models(
    *,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: int = 30,
) -> List[str]:
    resolved_api_key = DEFAULT_API_KEY if api_key is None else api_key
    resolved_api_url = (DEFAULT_API_URL if api_url is None else api_url).rstrip("/")
    headers: Dict[str, str] = {}
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    response = requests.get(f"{resolved_api_url}/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    response_json = response.json()
    models = response_json.get("data") or []
    return [str(item.get("id", "")) for item in models if item.get("id")]


def chat_completion(
    *,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    enable_thinking: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retry_max_tokens: int = DEFAULT_RETRY_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    resolved_api_key = DEFAULT_API_KEY if api_key is None else api_key
    resolved_api_url = (DEFAULT_API_URL if api_url is None else api_url).rstrip("/")
    requested_model = DEFAULT_MODEL if model is None else model
    attempted_models = [requested_model]
    fallback_triggered = False
    fallback_reason = None

    response = _request_chat_completion(
        messages=messages,
        api_key=resolved_api_key,
        api_url=resolved_api_url,
        model=requested_model,
        enable_thinking=enable_thinking,
        timeout=timeout,
        request_retries=request_retries,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    final_model = requested_model
    if not response.ok:
        error_summary = (
            f"Request failed after attempting models={attempted_models}, "
            f"fallback_triggered={fallback_triggered}, status={response.status_code}"
        )
        raise requests.HTTPError(error_summary, response=response)

    response_json = response.json()
    if (
        retry_max_tokens > max_tokens
        and _needs_retry_with_more_tokens(response_json)
    ):
        response = _request_chat_completion(
            messages=messages,
            api_key=resolved_api_key,
            api_url=resolved_api_url,
            model=final_model,
            enable_thinking=enable_thinking,
            timeout=timeout,
            request_retries=request_retries,
            max_tokens=retry_max_tokens,
            temperature=temperature,
        )
        if not response.ok:
            error_summary = (
                f"Retry request failed after attempting models={attempted_models}, "
                f"fallback_triggered={fallback_triggered}, status={response.status_code}"
            )
            raise requests.HTTPError(error_summary, response=response)
        response_json = response.json()

    return {
        "status_code": response.status_code,
        "api_url": resolved_api_url,
        "model": final_model,
        "fallback_model": None,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "attempted_models": attempted_models,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "request_retries": request_retries,
        "retry_max_tokens": retry_max_tokens,
        "assistant_text": extract_text(response_json),
        "reasoning_text": extract_reasoning_text(response_json),
        "finish_reason": extract_finish_reason(response_json),
        "response_json": response_json,
    }


def chat_completion_text(
    *,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    enable_thinking: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retry_max_tokens: int = DEFAULT_RETRY_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    result = chat_completion(
        messages=messages,
        api_key=api_key,
        api_url=api_url,
        model=model,
        fallback_model=fallback_model,
        enable_thinking=enable_thinking,
        timeout=timeout,
        request_retries=request_retries,
        max_tokens=max_tokens,
        retry_max_tokens=retry_max_tokens,
        temperature=temperature,
    )
    return result["assistant_text"]
