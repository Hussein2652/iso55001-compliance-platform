import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract first balanced JSON object
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return None


class ModelClient:
    """Thin abstraction for LLM + VLM with on-prem friendly fallbacks.

    Resolution order for LLM:
    1) HTTP OpenAI-compatible endpoint (env LLM_HTTP_BASE)
    2) Local HF model at LLM_LOCAL_PATH (optional; heavy)
    3) None (caller should fallback)

    Resolution order for VLM OCR:
    1) HTTP OpenAI-compatible vision endpoint (env VLM_HTTP_BASE)
    2) pytesseract local OCR (if available)
    3) None (caller should fallback)
    """

    def __init__(self) -> None:
        self.http_base = os.getenv('LLM_HTTP_BASE')
        self.http_api_key = os.getenv('LLM_HTTP_API_KEY')
        self.http_model = os.getenv('LLM_HTTP_MODEL', 'llama-3.1-8b-instruct')
        self.local_path = os.getenv('LLM_LOCAL_PATH', str(Path('models/llama-3.1-8b-instruct-gptq-int4').resolve()))
        self.vlm_http_base = os.getenv('VLM_HTTP_BASE')
        self.vlm_http_model = os.getenv('VLM_HTTP_MODEL', 'llama-3.2-vision')

    def generate_structured(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.3, max_new_tokens: int = 512, schema_hint: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        # Prefer local model when present, unless explicitly forcing HTTP
        if self.local_path and Path(self.local_path).exists():
            js = self._gen_local(system_prompt, user_prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            if js:
                return js
        if self.http_base:
            js = self._gen_http(system_prompt, user_prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            if js:
                return js
        return None

    def _gen_http(self, system_prompt: str, user_prompt: str, *, temperature: float, max_new_tokens: int) -> Optional[Dict[str, Any]]:
        try:
            import requests  # type: ignore
        except Exception:
            return None
        base = self.http_base.rstrip('/')
        url = f"{base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.http_api_key:
            headers["Authorization"] = f"Bearer {self.http_api_key}"
        body = {
            "model": self.http_model,
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
            if r.status_code != 200:
                return None
            data = r.json()
            txt = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            return _extract_json(txt)
        except Exception:
            return None

    def _gen_local(self, system_prompt: str, user_prompt: str, *, temperature: float, max_new_tokens: int) -> Optional[Dict[str, Any]]:
        # Very best-effort; requires transformers + a GPTQ loader. If unavailable, skip.
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            import torch  # type: ignore
        except Exception:
            return None
        try:
            tok = AutoTokenizer.from_pretrained(self.local_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(self.local_path, local_files_only=True, torch_dtype=getattr(torch, 'float16', None))
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
            inputs = tok(prompt, return_tensors='pt')
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=float(temperature))
            out = tok.decode(gen[0], skip_special_tokens=True)
            return _extract_json(out)
        except Exception:
            return None

    def vision_extract(self, image_path: str, *, instruction: str = "Extract structured fields", hints: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        # Try HTTP OpenAI-compatible vision
        if self.vlm_http_base:
            js = self._vision_http(image_path, instruction=instruction, hints=hints)
            if js:
                return js
        # Try local OCR
        js = self._vision_local_ocr(image_path)
        if js:
            return js
        return None

    def _vision_http(self, image_path: str, *, instruction: str, hints: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            import requests  # type: ignore
        except Exception:
            return None
        base = self.vlm_http_base.rstrip('/')
        url = f"{base}/v1/chat/completions"
        # Encode image as data URL if local path
        content: List[Dict[str, Any]] = []
        if image_path.startswith('http://') or image_path.startswith('https://'):
            content = [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_path}},
            ]
        else:
            try:
                p = Path(image_path)
                mime = 'image/png' if p.suffix.lower() == '.png' else 'image/jpeg'
                b64 = base64.b64encode(p.read_bytes()).decode('ascii')
                data_url = f"data:{mime};base64,{b64}"
                content = [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            except Exception:
                return None
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self.vlm_http_model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "max_tokens": 512,
        }
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
            if r.status_code != 200:
                return None
            data = r.json()
            txt = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            return _extract_json(txt) or {"text": txt}
        except Exception:
            return None

    def _vision_local_ocr(self, image_path: str) -> Optional[Dict[str, Any]]:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except Exception:
            return None
        try:
            txt = pytesseract.image_to_string(Image.open(image_path))
            return {"text": txt}
        except Exception:
            return None
