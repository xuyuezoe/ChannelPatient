"""ChatClient: OpenAI-compatible relay client with retry, disk cache, thinking switch, call counting."""
from __future__ import annotations
import hashlib, json, re, time
from pathlib import Path
from typing import Any
from scc.config import settings

NON_RETRY = (401, 402, 404)


class ChatClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, thinking: str = "off",
                 cache_dir: str | Path | None = None, temperature: float = 0.0, timeout: float = 120.0, max_retries: int = 6, use_cache: bool = True):
        import httpx
        from openai import OpenAI
        # YAML 会把裸的 off/on 解析成布尔值，这里统一成字符串
        thinking = "off" if thinking in (False, "off", None) else "on"
        self.model, self.thinking, self.temperature, self.max_retries = model, thinking, temperature, max_retries
        self.base_url = base_url or settings.base_url; self.api_key = api_key or settings.api_key
        self.cache_dir = Path(cache_dir) if cache_dir else settings.cache_dir
        self.use_cache = use_cache
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=httpx.Client(trust_env=False, timeout=timeout), max_retries=0)
        self.calls = 0; self.cache_hits = 0; self.tokens_in = 0; self.tokens_out = 0

    # pickling: drop the HTTP client and rebuild it on load (lets an env be saved between turns)
    def __getstate__(self):
        d = dict(self.__dict__); d.pop("client", None); return d

    def __setstate__(self, d):
        import httpx
        from openai import OpenAI
        self.__dict__.update(d)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=httpx.Client(trust_env=False, timeout=120.0), max_retries=0)

    # ------------------------------------------------------------------ cache
    def _key(self, messages, schema, temperature, max_tokens) -> str:
        payload = json.dumps({"m": self.model, "msgs": messages, "schema": schema, "t": temperature, "n": max_tokens, "think": self.thinking}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key):
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            try:
                return json.load(open(p))["text"]
            except Exception:
                return None
        return None

    def _cache_put(self, key, text):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"text": text, "model": self.model}, open(self.cache_dir / f"{key}.json", "w"), ensure_ascii=False)

    # ------------------------------------------------------------------ call
    def _raw(self, messages, max_tokens, temperature, schema, stop) -> str:
        kw: dict[str, Any] = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        if stop:
            kw["stop"] = stop
        if self.thinking == "off":
            kw["extra_body"] = {"thinking": {"type": "disabled"}}
        if schema is not None:
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}}
        last = None
        for i in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(**kw)
                self.calls += 1
                if getattr(r, "usage", None):
                    self.tokens_in += r.usage.prompt_tokens or 0; self.tokens_out += r.usage.completion_tokens or 0
                msg = r.choices[0].message
                text = msg.content or ""
                if not text.strip() and getattr(msg, "reasoning_content", None):
                    text = msg.reasoning_content
                return text
            except Exception as e:
                last = e
                status = getattr(e, "status_code", None)
                if status in NON_RETRY:
                    raise
                if schema is not None and i == 0 and ("response_format" in str(e) or status == 400):
                    kw.pop("response_format", None)      # relay may not support json_schema: fall back to prompt-only JSON
                    continue
                time.sleep(min(3 * (2 ** i), 60))
        raise RuntimeError(f"ChatClient failed after {self.max_retries} attempts: {last}")

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float | None = None, json_schema: dict | None = None, stop=None) -> str | dict:
        temperature = self.temperature if temperature is None else temperature
        msgs = list(messages)
        if json_schema is not None:
            instr = "Respond with a single JSON object matching this schema, no prose, no code fences:\n" + json.dumps(json_schema)
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + instr}
            else:
                msgs.insert(0, {"role": "system", "content": instr})
        key = self._key(msgs, json_schema, temperature, max_tokens)
        cached = self._cache_get(key) if (self.use_cache and temperature == 0) else None
        if cached is not None:
            self.cache_hits += 1
            text = cached
        else:
            text = self._raw(msgs, max_tokens, temperature, json_schema, stop)
            if self.use_cache and temperature == 0:
                self._cache_put(key, text)
        if json_schema is None:
            return text
        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> dict:
        t = re.sub(r"<(think|thinking|reasoning)>.*?</\1>", "", text, flags=re.S).strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
        try:
            return json.loads(t)
        except Exception:
            m = re.search(r"\{.*\}", t, re.S)
            if not m:
                raise ValueError(f"no JSON object in: {text[:200]!r}")
            return json.loads(m.group(0))

    def stats(self) -> dict:
        return {"model": self.model, "calls": self.calls, "cache_hits": self.cache_hits, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}
