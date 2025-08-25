# AGENTS

## Testing
Run the following commands before committing changes:

- `python -m py_compile config_manager.py services/llm.py services/vlm.py services/cloud.py services/baidu.py server.py node/translate_node.py`
- `node --check js/services/api.js js/services/interceptor.js js/modules/PromptAssistant.js js/modules/settings.js js/modules/apiConfigManager.js`

## Dependencies
Install Python dependencies via `pip install -r requirements.txt` when the dependency list changes.
