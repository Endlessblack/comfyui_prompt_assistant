import json
import os
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any, List, Callable
import asyncio
import re
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import httpx
from .error_util import format_api_error

class VisionService:
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
                config = config_manager.get_vision_config()
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
            timeout=httpx.Timeout(30.0)  # 视觉模型需要更长的超时时间
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
            
        # 创建客户端
        client = AsyncOpenAI(**kwargs)
        return client

    @staticmethod
    def _get_config() -> Dict[str, Any]:
        """获取视觉模型配置"""
        from ..config_manager import config_manager
        config = config_manager.get_vision_config()
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
        """验证并提取视觉模型返回的文本内容"""
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
            print(f"{PREFIX} [Gemini] raw_response: {VisionService._redact_api_key(raw)}")
        try:
            content = VisionService._extract_valid_content(resp, provider_display_name)
        except ValueError as e:
            if provider == 'gemini' and '空结果' in str(e):
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop('response_format', None)
                retry_kwargs['temperature'] = 0.3
                retry_kwargs['top_p'] = 1.0
                resp = await client.chat.completions.create(**retry_kwargs)
                raw_retry = resp.model_dump_json()
                print(f"{PREFIX} [Gemini] raw_response_retry: {VisionService._redact_api_key(raw_retry)}")
                content = VisionService._extract_valid_content(resp, provider_display_name)
            else:
                raise
        return content, resp

    @staticmethod
    def preprocess_image(image_data: str, request_id: Optional[str] = None) -> str:
        """
        预处理图像数据，包括压缩和调整大小
        
        参数:
            image_data: 图像数据（Base64编码或URL）
            request_id: 请求ID，用于日志记录
            
        返回:
            处理后的图像数据
        """
        from ..server import PREFIX
        
        try:
            # 检查是否为base64编码的图像数据
            if image_data.startswith('data:image'):
                # 提取base64数据
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                
                # 打开图像
                img = Image.open(BytesIO(image_bytes))
                original_size = img.size
                original_format = img.format or 'JPEG'
                original_bytes = len(image_bytes)
                
                # 调整大小，保持纵横比
                max_size = 1024  # 最大尺寸设为1024px
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                # 压缩图像
                buffer = BytesIO()
                save_format = 'JPEG' if original_format not in ['PNG', 'GIF'] else original_format
                
                # 根据格式选择保存参数
                if save_format == 'JPEG':
                    img.save(buffer, format=save_format, quality=85, optimize=True)
                elif save_format == 'PNG':
                    img.save(buffer, format=save_format, optimize=True, compress_level=7)
                else:
                    img.save(buffer, format=save_format)
                
                compressed_bytes = buffer.getvalue()
                
                # 转回base64
                compressed_b64 = base64.b64encode(compressed_bytes).decode('utf-8')
                processed_image_data = f"{header},{compressed_b64}"
                
                # 计算压缩比例
                compressed_size = len(compressed_bytes)
                compression_ratio = (1 - compressed_size / original_bytes) * 100
                
                # 记录日志
                print(f"{PREFIX} 图像预处理 | 请求ID:{request_id} | 原始尺寸:{original_size} | "
                      f"处理后尺寸:{img.size} | 压缩率:{compression_ratio:.1f}% | "
                      f"原始大小:{original_bytes/1024:.1f}KB | 压缩后:{compressed_size/1024:.1f}KB")
                
                return processed_image_data
            
            # 如果不是base64编码的图像数据，直接返回
            return image_data
            
        except Exception as e:
            from ..server import WARN_PREFIX
            print(f"{WARN_PREFIX} 图像预处理失败 | 请求ID:{request_id} | 错误:{str(e)}")
            # 预处理失败时返回原始图像数据
            return image_data

    @staticmethod
    async def analyze_image(image_data: str, request_id: Optional[str] = None,
                          stream_callback: Optional[Callable[[str], None]] = None,
                          prompt_content: Optional[str] = None,
                          custom_provider: Optional[str] = None,
                          custom_provider_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        使用视觉模型分析图像
        
        参数:
            image_data: 图像数据（Base64编码）
            request_id: 请求ID
            lang: 分析语言，zh为中文，en为英文
            stream_callback: 流式输出的回调函数
            custom_prompt: 自定义提示词
            custom_provider: 自定义提供商
            custom_provider_config: 自定义提供商配置
            
        返回:
            包含分析结果的字典
        """
        from ..server import PREFIX, ERROR_PREFIX
        try:
            # 获取配置
            if custom_provider and custom_provider_config:
                # 使用自定义提供商和配置
                provider = custom_provider
                api_key = custom_provider_config.get('api_key', '')
                model = custom_provider_config.get('model', '')
                base_url = custom_provider_config.get('base_url', '')
                temperature = custom_provider_config.get('temperature', 0.7)
                top_p = custom_provider_config.get('top_p', 0.9)
                max_tokens = custom_provider_config.get('max_tokens', 2000)
            else:
                # 使用默认配置
                config = VisionService._get_config()
                api_key = config.get('api_key')
                model = config.get('model')
                provider = config.get('provider', 'unknown')
                base_url = config.get('base_url')
                temperature = config.get('temperature', 0.7)
                top_p = config.get('top_p', 0.9)
                max_tokens = config.get('max_tokens', 2000)
            
            if not api_key:
                return {"success": False, "error": "请先配置视觉模型API密钥"}
            if not model:
                return {"success": False, "error": "未配置视觉模型名称"}
                
            # 检查图片数据格式
            if not image_data:
                return {"success": False, "error": "未提供图像数据"}
                
            # 处理图像数据格式
            if not image_data.startswith('data:image'):
                # 尝试添加前缀
                image_data = f"data:image/jpeg;base64,{image_data}"
            
            # 预处理图像数据（压缩和调整大小）
            image_data = VisionService.preprocess_image(image_data, request_id)
                
            # 获取提供商显示名称
            provider_display_name = {
                'zhipu': '智谱',
                'siliconflow': '硅基流动',
                'openai': 'OpenAI',
                'gemini': 'Gemini',
                'custom': '自定义'
            }.get(provider, provider)
            
            # 直接使用传入的提示词内容
            system_prompt = prompt_content
            if not system_prompt:
                return {"success": False, "error": "未提供有效的提示词内容"}
            
            # 发送请求
            print(f"{PREFIX} 调用视觉模型 | 服务:{provider_display_name} | 请求ID:{request_id} | 模型:{model}")
            
            # 使用OpenAI SDK
            client = VisionService.get_openai_client(api_key, provider, base_url)
            try:
                # 添加调试信息
                print(f"{PREFIX} 调用视觉模型API | 服务:{provider_display_name} | 模型:{model}")

                request_kwargs = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": image_data}}
                        ]
                    }],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "tool_choice": "none",
                }
                if provider != 'gemini':
                    request_kwargs["response_format"] = {"type": "text"}

                # Gemini 在 OpenAI 兼容流式模式下经常出现空增量，优先使用非流式，保留回调时再尝试流式
                if provider == 'gemini' and not stream_callback:
                    non_stream_kwargs = dict(request_kwargs)
                    non_stream_kwargs["stream"] = False
                    full_content, resp2 = await VisionService._extract_with_retry(
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
                        addition, resp3 = await VisionService._extract_with_retry(client, req3, provider, provider_display_name)
                        if addition:
                            full_content += addition
                    if not full_content.strip():
                        return {"success": False, "error": f"{provider_display_name}返回空结果"}
                else:
                    request_kwargs["stream"] = True
                    resp = await client.chat.completions.create(**request_kwargs)
                    full_content = ""
                    finish = None
                    first_logged = False
                    async for chunk in resp:
                        if provider == 'gemini':
                            print(f"{PREFIX} [Gemini|stream] raw_chunk: {VisionService._redact_api_key(chunk.model_dump_json())}")
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
                                full_content, resp2 = await VisionService._extract_with_retry(client, req2, provider, provider_display_name)
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
                                    addition, resp3 = await VisionService._extract_with_retry(client, req3, provider, provider_display_name)
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
                print(f"{PREFIX} 视觉模型分析成功 | 服务:{provider_display_name} | 请求ID:{request_id} | 结果字符数:{len(full_content)}")

                return {
                    "success": True,
                    "data": {
                        "description": full_content
                    }
                }
            except asyncio.CancelledError:
                print(f"{PREFIX} 视觉模型分析任务在服务层被取消 | ID:{request_id}")
                return {"success": False, "error": "请求已取消", "cancelled": True}
            except Exception as e:
                err = VisionService._redact_api_key(format_api_error(e, provider_display_name))
                return {"success": False, "error": err}
                
        except asyncio.CancelledError:
            print(f"{PREFIX} 视觉分析任务在服务层被取消 | ID:{request_id}")
            return {"success": False, "error": "请求已取消", "cancelled": True}
        except Exception as e:
            print(f"{ERROR_PREFIX} 视觉分析过程异常 | 错误:{str(e)}")
            return {"success": False, "error": VisionService._redact_api_key(str(e))}


if __name__ == "__main__":
    from openai.types.chat import ChatCompletion
    import asyncio

    def _make(data):
        base = {"id": "test", "model": "gpt"}
        base.update(data)
        return ChatCompletion.model_validate(base)

    case_a = {"choices":[{"message":{"role":"assistant","content":None},"finish_reason":"stop"}],"parts":[{"text":"Recovered"}]}
    assert VisionService._extract_valid_content(_make(case_a), "Gemini") == "Recovered"

    case_b = {"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"Hello"},{"type":"text","text":" world"}]},"finish_reason":"stop"}]}
    assert VisionService._extract_valid_content(_make(case_b), "Gemini") == "Hello world"

    case_c = {"choices":[{"message":{"role":"assistant","tool_calls":[{}]},"finish_reason":"stop"}],"parts":[{"text":"From parts"}]}
    assert VisionService._extract_valid_content(_make(case_c), "Gemini") == "From parts"

    empty_resp = {"choices":[{"message":{"role":"assistant","content":None},"finish_reason":"stop"}]}
    good_resp = {"choices":[{"message":{"role":"assistant","content":"Retry ok"},"finish_reason":"stop"}]}

    class DummyClient:
        def __init__(self):
            self.calls = 0
            self.chat = type("obj", (), {"completions": type("obj", (), {"create": self.create})()})

        async def create(self, **kwargs):
            self.calls += 1
            data = empty_resp if self.calls == 1 else good_resp
            return _make(data)

    async def _test_retry():
        content, _ = await VisionService._extract_with_retry(DummyClient(), {"model": "m"}, "gemini", "Gemini")
        assert content == "Retry ok"

    asyncio.run(_test_retry())
    print("self-test passed")

