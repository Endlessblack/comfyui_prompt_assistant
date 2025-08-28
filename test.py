#!/usr/bin/env python3
"""
Standalone tester for Gemini native requests (text and vision) matching the
current logic in services/llm.py and services/vlm.py, but runnable directly.

Usage examples (PowerShell):

  # Text
  $env:GEMINI_API_KEY='<YOUR_KEY>'
  python test.py text --model gemini-2.5-flash --system "Please answer in English." \
      --user "Say hello in one sentence." --verbose

  # Vision
  $env:GEMINI_API_KEY='<YOUR_KEY>'
  $b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes('C:\path\to\image.jpg'))
  $img="data:image/jpeg;base64,$b64"
  python test.py vision --model gemini-2.5-flash --system "Describe the image." \
      --image "$env:img" --verbose

Flags:
  --max-tries 0 means infinite retries (Ctrl+C to stop). Default 8.
  --base defaults to https://generativelanguage.googleapis.com/v1beta
  --log writes responses to logs/gemini_test_<timestamp>.log by default
"""

import argparse
import base64
import datetime as _dt
import json
import mimetypes
import os
import sys
from typing import Dict, Any, List, Optional

import httpx


def _ensure_logs_dir() -> str:
    d = os.path.join(os.getcwd(), 'logs')
    os.makedirs(d, exist_ok=True)
    return d


def _open_log_file(path: Optional[str]):
    if path:
        return open(path, 'a', encoding='utf-8')
    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    d = _ensure_logs_dir()
    p = os.path.join(d, f'gemini_test_{ts}.log')
    return open(p, 'a', encoding='utf-8')


def _extract_text_from_candidates(j: Dict[str, Any]) -> str:
    out: List[str] = []
    for c in (j.get('candidates') or []):
        content = (c or {}).get('content') or {}
        for p in (content.get('parts') or []):
            t = p.get('text') or p.get('output_text') or p.get('content')
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
    return ''.join(out).strip()


def _make_text_payload(system: str, user: str, max_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    return {
        'system_instruction': {'parts': [{'text': system or ''}]},
        'contents': [{
            'role': 'user',
            'parts': [{'text': user or ''}],
        }],
        'generation_config': {
            'temperature': temperature,
            'top_p': top_p,
            'max_output_tokens': max_tokens,
            'response_mime_type': 'text/plain',
        },
    }


def _data_url_from_path(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = 'image/jpeg'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f'data:{mime};base64,{b64}'


def _make_vision_payload(system: str, image_data_url: str, max_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []
    if system:
        parts.append({'text': system})
    mime = 'image/jpeg'
    data_b64 = ''
    if image_data_url.startswith('data:image'):
        try:
            header, enc = image_data_url.split(',', 1)
            if ';base64' in header:
                mime = header.split(':', 1)[1].split(';', 1)[0]
            data_b64 = enc
        except Exception:
            data_b64 = ''
    else:
        # Treat as filesystem path
        image_data_url = _data_url_from_path(image_data_url)
        header, enc = image_data_url.split(',', 1)
        if ';base64' in header:
            mime = header.split(':', 1)[1].split(';', 1)[0]
        data_b64 = enc
    if data_b64:
        parts.append({'inline_data': {'mime_type': mime, 'data': data_b64}})
    return {
        'contents': [{
            'role': 'user',
            'parts': parts,
        }],
        'generation_config': {
            'temperature': temperature,
            'top_p': top_p,
            'max_output_tokens': max_tokens,
            'response_mime_type': 'text/plain',
        },
    }


def _post_native(base: str, model: str, api_key: str, payload: Dict[str, Any], verbose: bool, logf) -> Dict[str, Any]:
    url = f"{base.rstrip('/')}/models/{model}:generateContent"
    print(f"HTTP Request: POST {url}")
    with httpx.Client(timeout=45.0) as s:
        r = s.post(url, headers={'x-goog-api-key': api_key}, json=payload)
        try:
            r.raise_for_status()
        except Exception as e:
            # Log body for diagnostics
            msg = f"HTTP {r.status_code} body: {r.text[:2000]}"
            print(msg)
            logf.write(msg + '\n')
            raise e
        j = r.json()
    if verbose:
        raw = json.dumps(j, ensure_ascii=False)
        line = f"[Gemini] raw_response: {raw}"
        print(line)
        logf.write(line + '\n')
    return j


def run_text(args) -> int:
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print('Missing API key. Provide --api-key or set GEMINI_API_KEY.')
        return 2
    tries = 0
    with _open_log_file(args.log) as logf:
        while True:
            tries += 1
            payload = _make_text_payload(args.system or '', args.user or '', args.max_tokens, args.temperature, args.top_p)
            j = _post_native(args.base, args.model, api_key, payload, args.verbose, logf)
            text = _extract_text_from_candidates(j)
            if text:
                print('TEXT=', text)
                logf.write('TEXT=' + text + '\n')
                return 0
            # Try increasing tokens if we hit MAX_TOKENS without text (hidden thoughts exhausted budget)
            finish = next(((c or {}).get('finishReason') for c in (j.get('candidates') or []) if isinstance(c, dict)), None)
            if finish == 'MAX_TOKENS':
                args.max_tokens = min(max(1024, args.max_tokens * 2), 4096)
                print(f"[Warn] MAX_TOKENS without text; bumping max_tokens to {args.max_tokens} and retrying...")
                logf.write(f"[Warn] MAX_TOKENS no text; bump max_tokens -> {args.max_tokens}\n")
            else:
                print(f"[Warn] Empty text on try #{tries}. Retrying...")
                logf.write(f"[Warn] Empty text on try #{tries}. Retrying...\n")
            if args.max_tries > 0 and tries >= args.max_tries:
                print('[Error] Reached max tries without content.')
                return 1


def run_vision(args) -> int:
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print('Missing API key. Provide --api-key or set GEMINI_API_KEY.')
        return 2
    tries = 0
    with _open_log_file(args.log) as logf:
        while True:
            tries += 1
            payload = _make_vision_payload(args.system or '', args.image, args.max_tokens, args.temperature, args.top_p)
            j = _post_native(args.base, args.model, api_key, payload, args.verbose, logf)
            text = _extract_text_from_candidates(j)
            if text:
                print('VISION=', text)
                logf.write('VISION=' + text + '\n')
                return 0
            finish = next(((c or {}).get('finishReason') for c in (j.get('candidates') or []) if isinstance(c, dict)), None)
            if finish == 'MAX_TOKENS':
                args.max_tokens = min(max(1024, args.max_tokens * 2), 4096)
                print(f"[Warn] MAX_TOKENS without text; bumping max_tokens to {args.max_tokens} and retrying...")
                logf.write(f"[Warn] MAX_TOKENS no text; bump max_tokens -> {args.max_tokens}\n")
            else:
                print(f"[Warn] Empty description on try #{tries}. Retrying...")
                logf.write(f"[Warn] Empty description on try #{tries}. Retrying...\n")
            if args.max_tries > 0 and tries >= args.max_tries:
                print('[Error] Reached max tries without content.')
                return 1


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description='Gemini native API tester (text/vision).')
    sub = p.add_subparsers(dest='mode', required=True)

    common = dict(
        add_help=False
    )
    # Text mode
    pt = sub.add_parser('text', parents=[], help='Text generation test')
    pt.add_argument('--api-key', default=None)
    pt.add_argument('--model', default='gemini-2.5-flash')
    pt.add_argument('--system', default='Please answer in English.')
    pt.add_argument('--user', default='Say hello in one sentence.')
    pt.add_argument('--max-tokens', type=int, default=512)
    pt.add_argument('--temperature', type=float, default=0.7)
    pt.add_argument('--top-p', type=float, default=0.9)
    pt.add_argument('--base', default='https://generativelanguage.googleapis.com/v1beta')
    pt.add_argument('--max-tries', type=int, default=8, help='0 for infinite')
    pt.add_argument('--verbose', action='store_true')
    pt.add_argument('--log', default=None, help='Log file path (default logs/gemini_test_<ts>.log)')

    # Vision mode
    pv = sub.add_parser('vision', parents=[], help='Vision generation test')
    pv.add_argument('--api-key', default=None)
    pv.add_argument('--model', default='gemini-2.5-flash')
    pv.add_argument('--system', default='Describe the image.')
    pv.add_argument('--image', required=True, help='Image path or data URL (data:image/...;base64,...)')
    pv.add_argument('--max-tokens', type=int, default=512)
    pv.add_argument('--temperature', type=float, default=0.7)
    pv.add_argument('--top-p', type=float, default=0.9)
    pv.add_argument('--base', default='https://generativelanguage.googleapis.com/v1beta')
    pv.add_argument('--max-tries', type=int, default=8, help='0 for infinite')
    pv.add_argument('--verbose', action='store_true')
    pv.add_argument('--log', default=None, help='Log file path (default logs/gemini_test_<ts>.log)')

    args = p.parse_args(argv)

    if args.mode == 'text':
        return run_text(args)
    elif args.mode == 'vision':
        return run_vision(args)
    else:
        print('Unknown mode')
        return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
