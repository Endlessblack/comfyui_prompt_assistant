import os
import random
import asyncio
import time
from typing import Optional

from google.cloud import translate

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
    async def translate_chunk(client: translate.TranslationServiceClient, chunk: str, project_id: str, location: str, from_lang: str, to_lang: str):
        parent = f"projects/{project_id}/locations/{location}"
        request = {
            "parent": parent,
            "contents": [chunk],
            "target_language_code": to_lang,
        }
        if from_lang != 'auto':
            request["source_language_code"] = from_lang
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.translate_text(request=request))
        translations = getattr(response, 'translations', [])
        translated_text = translations[0].translated_text.strip() if translations else ''
        if not translated_text:
            raise Exception("Cloud: 翻译结果为空")
        return translated_text

    @staticmethod
    async def translate(text: str, from_lang: str = 'auto', to_lang: str = 'zh', request_id: Optional[str] = None, is_auto: bool = False):
        try:
            from ..config_manager import config_manager
            config = config_manager.get_cloud_translate_config()
            project_id = config.get('project_id')
            location = config.get('location') or os.getenv('TRANSLATE_LOCATION') or 'global'
            credentials_path = config.get('credentials_path') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not credentials_path or not os.path.exists(credentials_path):
                return {"success": False, "error": "Cloud: 未检测到有效的GOOGLE_APPLICATION_CREDENTIALS"}
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') != credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            if not project_id:
                return {"success": False, "error": "Cloud: 请先配置项目ID"}

            client = translate.TranslationServiceClient()

            request_id = request_id or f"cloud_trans_{int(time.time())}_{random.randint(1000, 9999)}"
            from ..server import PREFIX, AUTO_TRANSLATE_PREFIX
            prefix = AUTO_TRANSLATE_PREFIX if is_auto else PREFIX
            print(f"{prefix} {'工作流自动翻译' if is_auto else '翻译请求'} | 服务:Cloud翻译 | 请求ID:{request_id} | 原文长度:{len(text)} | 方向:{from_lang}->{to_lang} | 位置:{location} | 方法:translateText")

            chunks = CloudTranslateService.split_text_by_paragraphs(text)
            if not chunks:
                chunks = [text]
            translated_parts = []
            for chunk in chunks:
                translated = await CloudTranslateService.translate_chunk(client, chunk, project_id, location, from_lang, to_lang)
                translated_parts.append(translated)
                if len(chunks) > 1:
                    await asyncio.sleep(1)
            translated_text = '\n'.join(translated_parts).strip()
            if not translated_text:
                return {"success": False, "error": "Cloud: 翻译结果为空"}
            print(f"{prefix} {'工作流翻译完成' if is_auto else '翻译完成'} | 服务:Cloud翻译 | 请求ID:{request_id} | 结果字符数:{len(translated_text)} | 位置:{location} | 方法:translateText")
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
            return {"success": False, "error": format_api_error(e, 'Cloud翻译')}

    @staticmethod
    async def batch_translate(texts: list, from_lang: str = 'auto', to_lang: str = 'zh'):
        tasks = [CloudTranslateService.translate(text, from_lang, to_lang) for text in texts]
        return await asyncio.gather(*tasks)

