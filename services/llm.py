import json
import os
import sys
from typing import Optional, Dict, Any, List, Callable
import asyncio
import re
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import httpx
from .error_util import format_api_error

class LLMService:
    _provider_base_urls = {
        'openai': None,  # 使用默认
        'siliconflow': 'https://api.siliconflow.cn/v1',
        'zhipu': 'https://open.bigmodel.cn/api/paas/v4',
        'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai',
        'custom': None  # 使用配置中的自定义URL
    }

    @classmethod
    def get_openai_client(
        cls, api_key: str, provider: str, base_url: Optional[str] = None
    ) -> AsyncOpenAI:
        """获取OpenAI客户端"""
        # 如果外部未提供base_url，则根据配置或预设获取
        if not base_url:
            if provider == 'custom':
                from ..config_manager import config_manager
                config = config_manager.get_llm_config()
                if 'providers' in config and 'custom' in config['providers']:
                    base_url = config['providers']['custom'].get('base_url')
                    # 确保base_url不以/chat/completions结尾，避免路径重复
                    if base_url and base_url.endswith('/chat/completions'):
                        base_url = base_url.rstrip('/chat/completions')
            else:
                base_url = cls._provider_base_urls.get(provider)
        
        # 创建简化的httpx客户端，不使用HTTP/2，避免额外依赖
        # Gemini 的 OpenAI 兼容端点推荐使用 Authorization: Bearer，不再追加 ?key 或 x-goog-api-key
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0 if provider == 'gemini' else 15.0)
        )

        kwargs = {
            "api_key": api_key,
            "http_client": http_client,
            "max_retries": 2  # 设置最大重试次数
        }
        if base_url:
            # 规范化 Gemini 的 OpenAI 兼容路径，确保带有 /openai 前缀
            if provider == 'gemini':
                # 去掉错误附带的结尾 /chat/completions，避免重复
                if base_url.endswith('/chat/completions'):
                    base_url = base_url[:-len('/chat/completions')]
                # 如果是官方域名但未携带 /openai，则追加一次
                if 'generativelanguage.googleapis.com' in base_url and '/openai' not in base_url:
                    base_url = base_url.rstrip('/') + '/openai'

            # 确保base_url末尾没有斜杠
            base_url = base_url.rstrip('/')
            kwargs["base_url"] = base_url

            # 添加调试日志
            from ..server import PREFIX
            print(f"{PREFIX} 创建OpenAI客户端 | 提供商:{provider} | 基础URL:{base_url}")

        # 创建客户端并缓存
        client = AsyncOpenAI(**kwargs)
        return client

    @staticmethod
    def _get_config() -> Dict[str, Any]:
        """获取LLM配置"""
        from ..config_manager import config_manager
        config = config_manager.get_llm_config()
        current_provider = config.get('provider')
        
        # 获取实际配置
        if 'providers' in config and current_provider in config['providers']:
            provider_config = config['providers'][current_provider]
            return {
                'provider': current_provider,
                'model': provider_config.get('model', ''),
                'base_url': provider_config.get('base_url', ''),
                'api_key': provider_config.get('api_key', ''),
                'temperature': provider_config.get('temperature', 0.7),
                'top_p': provider_config.get('top_p', 0.9),
                'max_tokens': provider_config.get('max_tokens', 2000)
            }
        else:
            # 兼容旧版配置格式
            return config

    @staticmethod
    def _redact_api_key(text: str) -> str:
        if not isinstance(text, str):
            return text
        return re.sub(r'(?<=\?key=)[^&]+', '***redacted***', text)

    @staticmethod
    def _extract_valid_content(completion: ChatCompletion, provider_display_name: str) -> str:
        """验证并提取模型返回的文本内容"""
        if not isinstance(completion, ChatCompletion):
            raise ValueError(f"{provider_display_name}返回无效响应")
        if not getattr(completion, 'id', None) or not getattr(completion, 'model', None):
            raise ValueError(f"{provider_display_name}返回结构不完整")
        choices = getattr(completion, 'choices', None)
        if not choices or len(choices) == 0:
            raise ValueError(f"{provider_display_name}无可用结果")
        choice = choices[0]
        message = getattr(choice, 'message', None)
        if not message:
            raise ValueError(f"{provider_display_name}响应缺少消息")
        if getattr(message, 'refusal', None):
            raise ValueError(f"{provider_display_name}拒绝提供内容")
        if getattr(choice, 'blocked', False) or getattr(choice, 'blocked_reason', None):
            raise ValueError(f"{provider_display_name}返回被阻止的内容")
        content = getattr(message, 'content', None)
        text = ""
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            buf: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text") or ""
                    if isinstance(t, str) and t.strip():
                        buf.append(t.strip())
                elif isinstance(part, str):
                    if part.strip():
                        buf.append(part.strip())
            text = "".join(buf).strip()

        def _search(obj: Any) -> Optional[str]:
            """在兼容响应中尽量定位文本，但避免误把 id、model 等字符串当作正文。"""
            if isinstance(obj, dict):
                # 优先常见文本字段
                for key in ["text", "output_text", "value"]:
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                # 兼容 content:list 的内层
                val = obj.get("content")
                if isinstance(val, list):
                    buf: List[str] = []
                    for p in val:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("content")
                            if isinstance(t, str) and t.strip():
                                buf.append(t.strip())
                    if buf:
                        return "".join(buf)
                # 继续深搜其它字段
                for k, v in obj.items():
                    # 跳过明显不是正文的字段，避免误判
                    if k in {"id", "model", "object", "system_fingerprint"}:
                        continue
                    found = _search(v)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = _search(item)
                    if found:
                        return found
            # 不再返回任意字符串，避免将 id 等误当正文
            return None

        if not text:
            raw = completion.model_dump()
            text = _search(raw)

        if text:
            return text.strip()
        raise ValueError(f"{provider_display_name}返回空结果")

    @staticmethod
    async def _extract_with_retry(client, request_kwargs: Dict[str, Any], provider: str, provider_display_name: str):
        resp = await client.chat.completions.create(**request_kwargs)
        if provider == 'gemini':
            from ..server import PREFIX
            raw = resp.model_dump_json()
            print(f"{PREFIX} [Gemini] raw_response: {LLMService._redact_api_key(raw)}")
        try:
            content = LLMService._extract_valid_content(resp, provider_display_name)
        except ValueError as e:
            if provider == 'gemini' and '空结果' in str(e):
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop('response_format', None)
                retry_kwargs['temperature'] = 0.3
                retry_kwargs['top_p'] = 1.0
                resp = await client.chat.completions.create(**retry_kwargs)
                raw_retry = resp.model_dump_json()
                print(f"{PREFIX} [Gemini] raw_response_retry: {LLMService._redact_api_key(raw_retry)}")
                content = LLMService._extract_valid_content(resp, provider_display_name)
            else:
                raise
        return content, resp
    
    @staticmethod
    async def expand_prompt(prompt: str, request_id: Optional[str] = None, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        使用大语言模型扩写提示词，自动判断用户输入语言，并设置大语言模型回答语言。
        支持流式输出以提高响应速度。
        
        参数:
            prompt: 要扩写的提示词
            request_id: 请求ID
            stream_callback: 流式输出的回调函数
            
        返回:
            包含扩写结果的字典
        """
        try:
            # 获取配置
            config = LLMService._get_config()
            api_key = config.get('api_key')
            model = config.get('model')
            provider = config.get('provider', 'unknown')
            base_url = config.get('base_url')
            temperature = config.get('temperature', 0.7)
            top_p = config.get('top_p', 0.9)
            max_tokens = config.get('max_tokens', 2000)
            
            if not api_key:
                return {"success": False, "error": "请先配置大语言模型 API密钥"}
            if not model:
                return {"success": False, "error": "未配置模型名称"}

            # 从server.py导入颜色常量和前缀
            from ..server import PREFIX, ERROR_PREFIX
            
            # 获取提供商显示名称
            provider_display_name = {
                'zhipu': '智谱',
                'siliconflow': '硅基流动',
                'openai': 'OpenAI',
                'gemini': 'Gemini',
                'custom': '自定义'
            }.get(provider, provider)
            
            print(f"{PREFIX} LLM扩写请求 | 服务:{provider_display_name} | ID:{request_id} | 内容:{prompt[:30]}...")

            # 加载系统提示词
            from ..config_manager import config_manager
            system_prompts = config_manager.get_system_prompts()
            
            if not system_prompts or 'expand_prompts' not in system_prompts:
                return {"success": False, "error": "扩写系统提示词加载失败"}
            
            # 获取激活的提示词ID
            active_prompt_id = system_prompts.get('active_prompts', {}).get('expand', 'expand_default')
            
            # 获取对应的提示词
            if active_prompt_id not in system_prompts['expand_prompts']:
                # 如果找不到激活的提示词，尝试使用第一个可用的提示词
                if len(system_prompts['expand_prompts']) > 0:
                    active_prompt_id = list(system_prompts['expand_prompts'].keys())[0]
                else:
                    return {"success": False, "error": "未找到可用的扩写系统提示词"}
            
            system_message = system_prompts['expand_prompts'][active_prompt_id]
            
            # 输出使用的提示词名称
            prompt_name = system_message.get('name', active_prompt_id)
            print(f"{PREFIX} 使用扩写提示词: {prompt_name} | ID:{active_prompt_id}")

            # 判断用户输入语言
            def is_chinese(text):
                return any('\u4e00' <= char <= '\u9fff' for char in text)
            if is_chinese(prompt):
                lang_message = {"role": "system", "content": "请用中文回答"}
            else:
                lang_message = {"role": "system", "content": "Please answer in English."}

            # 构建消息
            messages = [
                lang_message,
                system_message,
                {"role": "user", "content": prompt}
            ]

            # 使用OpenAI SDK
            client = LLMService.get_openai_client(api_key, provider, base_url)
            try:
                # 添加调试信息
                print(f"{PREFIX} 调用LLM API | 服务:{provider_display_name} | 模型:{model}")

                request_kwargs = {
                    "model": model,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "tool_choice": "none",
                }
                if provider != 'gemini':
                    request_kwargs["response_format"] = {"type": "text"}

                # Gemini 在 OpenAI 兼容流式模式下经常出现空增量，优先使用非流式，保留回调时再尝试流式
                if provider == 'gemini' and not stream_callback:
                    non_stream_kwargs = dict(request_kwargs)
                    non_stream_kwargs["stream"] = False
                    full_content, resp2 = await LLMService._extract_with_retry(
                        client, non_stream_kwargs, provider, provider_display_name
                    )
                    finish = getattr(resp2.choices[0], "finish_reason", None) if resp2 else None
                    # 如果因为长度截断，追加一次续写请求（避免多次重试）
                    if finish == "length" and full_content:
                        tail_or_continued = full_content[-1200:]
                        cont_messages = request_kwargs["messages"] + [
                            {"role": "assistant", "content": tail_or_continued},
                            {"role": "user", "content": "Continue."}
                        ]
                        tmp_max_tokens = min(int(max_tokens * 1.5) if max_tokens else 1500, 4096)
                        req3 = {
                            "model": model,
                            "messages": cont_messages,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": tmp_max_tokens,
                            "stream": False,
                            "tool_choice": "none",
                        }
                        addition, resp3 = await LLMService._extract_with_retry(client, req3, provider, provider_display_name)
                        if addition:
                            full_content += addition
                    if not full_content.strip():
                        return {"success": False, "error": f"{provider_display_name}返回空结果"}
                else:
                    request_kwargs["stream"] = True
                    stream = await client.chat.completions.create(**request_kwargs)
                    full_content = ""
                    finish = None
                    first_logged = False
                    async for chunk in stream:
                        if provider == 'gemini':
                            print(f"{PREFIX} [Gemini|stream] raw_chunk: {LLMService._redact_api_key(chunk.model_dump_json())}")
                        choice0 = chunk.choices[0]
                        delta = getattr(choice0, "delta", None)
                        finish = getattr(choice0, "finish_reason", finish)
                        part = getattr(delta, "content", None) if delta else None
                        piece = ""
                        if isinstance(part, str):
                            piece = part
                        elif isinstance(part, list):
                            buf: List[str] = []
                            for p in part:
                                if isinstance(p, dict):
                                    t = p.get("text") or p.get("content") or ""
                                    if isinstance(t, str) and t:
                                        buf.append(t)
                            piece = "".join(buf)
                        if piece:
                            if not first_logged and provider == 'gemini':
                                print(f"{PREFIX} [Gemini|stream] first-chunk len={len(piece)}")
                                first_logged = True
                            full_content += piece
                            if stream_callback:
                                stream_callback(piece)
                    if provider == 'gemini':
                        print(f"{PREFIX} [Gemini|stream] finish_reason={finish} total_len={len(full_content)}")
                        if not full_content:
                            req2 = dict(request_kwargs)
                            req2["stream"] = False
                            try:
                                full_content, resp2 = await LLMService._extract_with_retry(client, req2, provider, provider_display_name)
                            except ValueError:
                                full_content = ""
                                resp2 = None
                            finish = getattr(resp2.choices[0], "finish_reason", finish) if resp2 else finish
                            print(f"{PREFIX} [Gemini|stream] fallback non-stream used len={len(full_content)}")
                        if finish == "length" or not full_content:
                            # 将自动续写次数限制为 1，避免多次请求
                            for k in range(1):
                                tail_or_continued = full_content[-1200:] if full_content else "(continued)"
                                cont_messages = request_kwargs["messages"] + [
                                    {"role": "assistant", "content": tail_or_continued},
                                    {"role": "user", "content": "Continue."}
                                ]
                                tmp_max_tokens = min(int(max_tokens * 1.5) if max_tokens else 1500, 4096)
                                req3 = {
                                    "model": model,
                                    "messages": cont_messages,
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "max_tokens": tmp_max_tokens,
                                    "stream": False,
                                    "tool_choice": "none",
                                }
                                if provider != 'gemini':
                                    req3["response_format"] = {"type": "text"}
                                try:
                                    addition, resp3 = await LLMService._extract_with_retry(client, req3, provider, provider_display_name)
                                except ValueError:
                                    addition = ""
                                    resp3 = None
                                finish = getattr(resp3.choices[0], "finish_reason", finish) if resp3 else finish
                                if addition:
                                    full_content += addition
                                print(f"{PREFIX} auto-continue#{k+1} finish={finish} len={len(full_content)}")
                                if finish != "length" or not addition:
                                    break
                        if finish in {"content_filter", "safety"}:
                            print(f"{PREFIX} finish_reason={finish} total_len={len(full_content)}")
                        elif not full_content.strip():
                            return {"success": False, "error": f"{provider_display_name}返回空结果"}
                    else:
                        if not full_content.strip():
                            return {"success": False, "error": f"{provider_display_name}返回空结果"}

                # 输出结构化成功日志
                print(f"{PREFIX} LLM扩写成功 | 服务:{provider_display_name} | 请求ID:{request_id} | 结果字符数:{len(full_content)}")

                return {
                    "success": True,
                    "data": {
                        "original": prompt,
                        "expanded": full_content
                    }
                }
            except asyncio.CancelledError:
                print(f"{PREFIX} LLM扩写任务在服务层被取消 | ID:{request_id}")
                return {"success": False, "error": "请求已取消", "cancelled": True}
            except Exception as e:
                err = LLMService._redact_api_key(format_api_error(e, provider_display_name))
                return {"success": False, "error": err}
                
        except asyncio.CancelledError:
            from ..server import PREFIX
            print(f"{PREFIX} LLM扩写任务在服务层被取消 | ID:{request_id}")
            return {"success": False, "error": "请求已取消", "cancelled": True}
        except Exception as e:
            # 从server.py导入颜色常量和前缀
            from ..server import ERROR_PREFIX
            print(f"{ERROR_PREFIX} LLM扩写请求失败 | 错误:{str(e)}")
            return {"success": False, "error": LLMService._redact_api_key(str(e))}
    
    @staticmethod
    async def translate(text: str, from_lang: str = 'auto', to_lang: str = 'zh', request_id: Optional[str] = None, is_auto: bool = False, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        使用大语言模型翻译文本，自动设置提示词语言和输出语言。
        支持流式输出以提高响应速度。
        
        参数:
            text: 要翻译的文本
            from_lang: 源语言 (默认为auto自动检测)
            to_lang: 目标语言 (默认为zh中文)
            request_id: 请求ID
            is_auto: 是否为工作流自动翻译
            stream_callback: 流式输出的回调函数
            
        返回:
            包含翻译结果的字典
        """
        try:
            # 获取配置
            config = LLMService._get_config()
            api_key = config.get('api_key')
            model = config.get('model')
            provider = config.get('provider', 'unknown')
            base_url = config.get('base_url')
            temperature = config.get('temperature', 0.7)
            top_p = config.get('top_p', 0.9)
            max_tokens = config.get('max_tokens', 2000)
            
            if not api_key:
                return {"success": False, "error": "请先配置大语言模型 API密钥"}
            if not model:
                return {"success": False, "error": "未配置模型名称"}

            # 从server.py导入颜色常量和前缀
            from ..server import PREFIX, AUTO_TRANSLATE_PREFIX
            
            # 获取提供商显示名称
            provider_display_name = {
                'zhipu': '智谱',
                'siliconflow': '硅基流动',
                'openai': 'OpenAI',
                'gemini': 'Gemini',
                'custom': '自定义'
            }.get(provider, provider)
            
            # 使用统一的前缀
            prefix = AUTO_TRANSLATE_PREFIX if is_auto else PREFIX
            print(f"{prefix} {'工作流自动翻译' if is_auto else '翻译请求'} | 服务:{provider_display_name}翻译 | 请求ID:{request_id} | 原文长度:{len(text)} | 方向:{from_lang}->{to_lang}")

            # 加载系统提示词
            from ..config_manager import config_manager
            system_prompts = config_manager.get_system_prompts()
            
            if not system_prompts or 'translate_prompts' not in system_prompts or 'ZH' not in system_prompts['translate_prompts']:
                return {"success": False, "error": "翻译系统提示词加载失败"}
            
            system_message = system_prompts['translate_prompts']['ZH']

            # 动态替换提示词中的{src_lang}和{dst_lang}
            lang_map = {'zh': '中文', 'en': '英文', 'auto': '原文'}
            src_lang = lang_map.get(from_lang, from_lang)
            dst_lang = lang_map.get(to_lang, to_lang)
            sys_msg_content = system_message['content'].replace('{src_lang}', src_lang).replace('{dst_lang}', dst_lang)
            sys_msg = {"role": "system", "content": sys_msg_content}

            # 设置输出语言
            if to_lang == 'en':
                lang_message = {"role": "system", "content": "Please answer in English."}
            else:
                lang_message = {"role": "system", "content": "请用中文回答"}

            # 构建消息
            messages = [
                lang_message,
                sys_msg,
                {"role": "user", "content": text}
            ]

            # 使用OpenAI SDK
            client = LLMService.get_openai_client(api_key, provider, base_url)
            try:
                # 添加调试信息
                print(f"{PREFIX} 调用LLM API | 服务:{provider_display_name} | 模型:{model}")

                request_kwargs = {
                    "model": model,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "tool_choice": "none",
                }
                if provider != 'gemini':
                    request_kwargs["response_format"] = {"type": "text"}

                request_kwargs["stream"] = True
                stream = await client.chat.completions.create(**request_kwargs)
                full_content = ""
                finish = None
                first_logged = False
                async for chunk in stream:
                    if provider == 'gemini':
                        print(f"{PREFIX} [Gemini|stream] raw_chunk: {LLMService._redact_api_key(chunk.model_dump_json())}")
                    choice0 = chunk.choices[0]
                    delta = getattr(choice0, "delta", None)
                    finish = getattr(choice0, "finish_reason", finish)
                    part = getattr(delta, "content", None) if delta else None
                    piece = ""
                    if isinstance(part, str):
                        piece = part
                    elif isinstance(part, list):
                        buf: List[str] = []
                        for p in part:
                            if isinstance(p, dict):
                                t = p.get("text") or p.get("content") or ""
                                if isinstance(t, str) and t:
                                    buf.append(t)
                        piece = "".join(buf)
                    if piece:
                        if not first_logged and provider == 'gemini':
                            print(f"{PREFIX} [Gemini|stream] first-chunk len={len(piece)}")
                            first_logged = True
                        full_content += piece
                        if stream_callback:
                            stream_callback(piece)
                if provider == 'gemini':
                    print(f"{PREFIX} [Gemini|stream] finish_reason={finish} total_len={len(full_content)}")
                    if not full_content:
                        req2 = dict(request_kwargs)
                        req2["stream"] = False
                        try:
                            full_content, resp2 = await LLMService._extract_with_retry(client, req2, provider, provider_display_name)
                        except ValueError:
                            full_content = ""
                            resp2 = None
                        finish = getattr(resp2.choices[0], "finish_reason", finish) if resp2 else finish
                        print(f"{PREFIX} [Gemini|stream] fallback non-stream used len={len(full_content)}")
                    if finish == "length" or not full_content:
                        for k in range(3):
                            tail_or_continued = full_content[-1200:] if full_content else "(continued)"
                            cont_messages = request_kwargs["messages"] + [
                                {"role": "assistant", "content": tail_or_continued},
                                {"role": "user", "content": "Continue."}
                            ]
                            tmp_max_tokens = min(int(max_tokens * 1.5) if max_tokens else 1500, 4096)
                            req3 = {
                                "model": model,
                                "messages": cont_messages,
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_tokens": tmp_max_tokens,
                                "stream": False,
                                "tool_choice": "none",
                            }
                            if provider != 'gemini':
                                req3["response_format"] = {"type": "text"}
                            try:
                                addition, resp3 = await LLMService._extract_with_retry(client, req3, provider, provider_display_name)
                            except ValueError:
                                addition = ""
                                resp3 = None
                            finish = getattr(resp3.choices[0], "finish_reason", finish) if resp3 else finish
                            if addition:
                                full_content += addition
                            print(f"{PREFIX} auto-continue#{k+1} finish={finish} len={len(full_content)}")
                            if finish != "length" or not addition:
                                break
                    if finish in {"content_filter", "safety"}:
                        print(f"{PREFIX} finish_reason={finish} total_len={len(full_content)}")
                    elif not full_content.strip():
                        return {"success": False, "error": f"{provider_display_name}返回空结果"}
                else:
                    if not full_content.strip():
                        return {"success": False, "error": f"{provider_display_name}返回空结果"}

                # 输出结构化成功日志
                prefix = AUTO_TRANSLATE_PREFIX if is_auto else PREFIX
                print(f"{prefix} {'工作流翻译完成' if is_auto else '翻译完成'} | 服务:{provider_display_name}翻译 | 请求ID:{request_id} | 结果字符数:{len(full_content)}")

                return {
                    "success": True,
                    "data": {
                        "from": from_lang,
                        "to": to_lang,
                        "original": text,
                        "translated": full_content
                    }
                }
            except asyncio.CancelledError:
                print(f"{prefix} {'工作流翻译' if is_auto else '翻译'}任务在服务层被取消 | ID:{request_id}")
                return {"success": False, "error": "请求已取消", "cancelled": True}
            except Exception as e:
                err = LLMService._redact_api_key(format_api_error(e, provider_display_name))
                return {"success": False, "error": err}
                
        except asyncio.CancelledError:
            from ..server import PREFIX, AUTO_TRANSLATE_PREFIX
            prefix = AUTO_TRANSLATE_PREFIX if is_auto else PREFIX
            print(f"{prefix} {'工作流翻译' if is_auto else '翻译'}任务在服务层被取消 | ID:{request_id}")
            return {"success": False, "error": "请求已取消", "cancelled": True}
        except Exception as e:
            return {"success": False, "error": LLMService._redact_api_key(str(e))}


if __name__ == "__main__":
    from openai.types.chat import ChatCompletion
    import asyncio

    def _make(data):
        base = {"id": "test", "model": "gpt"}
        base.update(data)
        return ChatCompletion.model_validate(base)

    # Case A (content:null with text elsewhere)
    case_a = {"choices":[{"message":{"role":"assistant","content":None},"finish_reason":"stop"}],"parts":[{"text":"Recovered"}]}
    assert LLMService._extract_valid_content(_make(case_a), "Gemini") == "Recovered"

    # Case B (content:list)
    case_b = {"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"Hello"},{"type":"text","text":" world"}]},"finish_reason":"stop"}]}
    assert LLMService._extract_valid_content(_make(case_b), "Gemini") == "Hello world"

    # Case C (tool_calls without text)
    case_c = {"choices":[{"message":{"role":"assistant","tool_calls":[{}]},"finish_reason":"stop"}],"parts":[{"text":"From parts"}]}
    assert LLMService._extract_valid_content(_make(case_c), "Gemini") == "From parts"

    empty_resp = {"choices":[{"message":{"role":"assistant","content":None},"finish_reason":"stop"}]}
    good_resp = {"choices":[{"message":{"role":"assistant","content":"Retry success"},"finish_reason":"stop"}]}

    class DummyClient:
        def __init__(self):
            self.calls = 0
            self.chat = type("obj", (), {"completions": type("obj", (), {"create": self.create})()})

        async def create(self, **kwargs):
            self.calls += 1
            data = empty_resp if self.calls == 1 else good_resp
            return _make(data)

    async def _test_retry():
        content, _ = await LLMService._extract_with_retry(DummyClient(), {"model": "m"}, "gemini", "Gemini")
        assert content == "Retry success"

    asyncio.run(_test_retry())
    print("self-test passed")
