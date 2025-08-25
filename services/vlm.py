import json
import os
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any, List, Callable
import asyncio
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
        # 视觉模型需要较长的超时时间
        if provider == 'gemini':
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                params={"key": api_key}
            )
        else:
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0)  # 视觉模型需要更长的超时时间
            )
        
        kwargs = {
            "api_key": api_key,
            "http_client": http_client,
            "max_retries": 2  # 设置最大重试次数
        }
        if base_url:
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
        finish_reason = getattr(choice, 'finish_reason', None)
        if finish_reason and finish_reason not in ('stop', 'length'):
            raise ValueError(f"{provider_display_name}响应异常: finish_reason={finish_reason}")
        message = getattr(choice, 'message', None)
        if not message:
            raise ValueError(f"{provider_display_name}响应缺少消息")
        if getattr(message, 'refusal', None):
            raise ValueError(f"{provider_display_name}拒绝提供内容")
        if getattr(choice, 'blocked', False) or getattr(choice, 'blocked_reason', None):
            raise ValueError(f"{provider_display_name}返回被阻止的内容")
        if getattr(message, 'tool_calls', None) and not getattr(message, 'content', None):
            raise ValueError(f"{provider_display_name}返回工具调用而无文本内容")
        content = getattr(message, 'content', '')
        if content is None:
            raise ValueError(f"{provider_display_name}返回空结果")
        # Gemini可能返回内容片段数组，需要合并
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not isinstance(content, str):
            raise ValueError(f"{provider_display_name}返回空结果")
        content = content.strip()
        if not content:
            raise ValueError(f"{provider_display_name}返回空结果")
        return content

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
                    "response_format": {"type": "text"},
                }

                request_kwargs["stream"] = True
                resp = await client.chat.completions.create(**request_kwargs)
                full_content = ""
                finish = None
                first_logged = False
                async for chunk in resp:
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
                        resp2 = await client.chat.completions.create(**req2)
                        try:
                            full_content = VisionService._extract_valid_content(resp2, provider_display_name)
                        except ValueError:
                            full_content = ""
                        finish = getattr(resp2.choices[0], "finish_reason", finish)
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
                                "response_format": {"type": "text"},
                                "stream": False,
                            }
                            resp3 = await client.chat.completions.create(**req3)
                            try:
                                addition = VisionService._extract_valid_content(resp3, provider_display_name)
                            except ValueError:
                                addition = ""
                            finish = getattr(resp3.choices[0], "finish_reason", finish)
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
                return {"success": False, "error": format_api_error(e, provider_display_name)}
                
        except asyncio.CancelledError:
            print(f"{PREFIX} 视觉分析任务在服务层被取消 | ID:{request_id}")
            return {"success": False, "error": "请求已取消", "cancelled": True}
        except Exception as e:
            print(f"{ERROR_PREFIX} 视觉分析过程异常 | 错误:{str(e)}")
            return {"success": False, "error": str(e)}
