import random
import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any

from .error_util import format_api_error


class CloudTranslateService:
    @staticmethod
    def split_text_by_paragraphs(text: str, max_length: int = 2000):
        if not text:
            return []
        lines = text.split('\n')
        chunks = []
        current = ''
        for line in lines:
            if len(line) > max_length:
                if current:
                    chunks.append(current)
                    current = ''
                remaining = line
                while remaining:
                    chunks.append(remaining[:max_length])
                    remaining = remaining[max_length:]
            elif current and len(current) + len(line) + 1 > max_length:
                chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    async def translate_chunk(session: aiohttp.ClientSession, chunk: str, api_key: str, from_lang: str, to_lang: str):
        url = 'https://translation.googleapis.com/language/translate/v2'
        params = {
            'q': chunk,
            'target': to_lang,
            'format': 'text',
            'key': api_key
        }
        if from_lang != 'auto':
            params['source'] = from_lang
        async with session.post(url, data=params, timeout=10) as resp:
            if resp.status != 200:
                raise Exception(f"Cloud: HTTP请求失败，状态码: {resp.status}")
            data = await resp.json()
            if 'error' in data:
                message = data['error'].get('message', '未知错误')
                raise Exception(f"Cloud: {message}")
            translations = data.get('data', {}).get('translations', [])
            translated_parts = [t.get('translatedText', '') for t in translations]
            translated_text = '\n'.join(translated_parts).strip()
            if not translated_text:
                raise Exception("Cloud: 翻译结果为空")
            return translated_text

    @staticmethod
    async def translate(text: str, from_lang: str = 'auto', to_lang: str = 'zh', request_id: Optional[str] = None, is_auto: bool = False):
        try:
            from ..config_manager import config_manager
            config = config_manager.get_cloud_translate_config()
            api_key = config.get('api_key')
            if not api_key:
                return {"success": False, "error": "Cloud: 请先配置Cloud Translate API密钥"}

            request_id = request_id or f"cloud_trans_{int(time.time())}_{random.randint(1000, 9999)}"
            from ..server import PREFIX, AUTO_TRANSLATE_PREFIX
            prefix = AUTO_TRANSLATE_PREFIX if is_auto else PREFIX
            print(f"{prefix} {'工作流自动翻译' if is_auto else '翻译请求'} | 服务:Cloud翻译 | 请求ID:{request_id} | 原文长度:{len(text)} | 方向:{from_lang}->{to_lang}")

            chunks = CloudTranslateService.split_text_by_paragraphs(text)
            if not chunks:
                chunks = [text]
            translated_parts = []
            async with aiohttp.ClientSession(trust_env=False) as session:
                for chunk in chunks:
                    translated = await CloudTranslateService.translate_chunk(session, chunk, api_key, from_lang, to_lang)
                    translated_parts.append(translated)
                    if len(chunks) > 1:
                        await asyncio.sleep(1)
            translated_text = '\n'.join(translated_parts).strip()
            if not translated_text:
                return {"success": False, "error": "Cloud: 翻译结果为空"}
            print(f"{prefix} {'工作流翻译完成' if is_auto else '翻译完成'} | 服务:Cloud翻译 | 请求ID:{request_id} | 结果字符数:{len(translated_text)}")
            return {
                "success": True,
                "data": {
                    "from": from_lang,
                    "to": to_lang,
                    "original": text,
                    "translated": translated_text
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def batch_translate(texts: list, from_lang: str = 'auto', to_lang: str = 'zh'):
        tasks = [CloudTranslateService.translate(text, from_lang, to_lang) for text in texts]
        return await asyncio.gather(*tasks)
