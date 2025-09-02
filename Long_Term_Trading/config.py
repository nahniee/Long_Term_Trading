from decouple import Config
from decouple import RepositoryEnv

class EnvConfig:
    def __init__(self, env_path):
        config = Config(RepositoryEnv(env_path))
        self.openai_key = config('OPENAI_API_KEY')