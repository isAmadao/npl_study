import os
import yaml

class Settings:
    def __init__(self, config_path: str = "config.yml"):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.api_host = self._get('api.host', '0.0.0.0')
        self.api_port = self._get('api.port', 8000)
        self.api_debug = self._get('api.debug', True)
        
        self.llm_provider = self._get('llm.provider', 'dashscope')
        self.llm_api_key = self._get('llm.api_key', '')
        self.llm_api_base = self._get('llm.api_base', '')
        self.llm_model = self._get('llm.model', 'qwen-turbo')
        self.llm_max_tokens = self._get('llm.max_tokens', 2048)
        self.llm_temperature = self._get('llm.temperature', 0.7)
        
        self.multimodal_provider = self._get('multimodal.provider', 'dashscope')
        self.multimodal_api_key = self._get('multimodal.api_key', '')
        self.multimodal_api_base = self._get('multimodal.api_base', '')
        self.multimodal_model = self._get('multimodal.model', 'qwen-vl-plus')
        self.multimodal_max_tokens = self._get('multimodal.max_tokens', 2048)
        self.multimodal_temperature = self._get('multimodal.temperature', 0.7)
        
        self.embedding_provider = self._get('embedding.provider', 'dashscope')
        self.embedding_api_key = self._get('embedding.api_key', '')
        self.embedding_api_base = self._get('embedding.api_base', '')
        self.embedding_model = self._get('embedding.model', 'text-embedding-v1')
        self.embedding_dimension = self._get('embedding.dimension', 1024)
    
    def _get(self, path: str, default=None):
        keys = path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key)
            if value is None:
                return default
        return value

settings = Settings()
