import os
from tomllib import load  # Python 3.11+ 内置模块

def detect_config_exist():
    if not os.path.exists("config.toml"):
        raise FileNotFoundError("config.toml not found, please create it first")


def load_apikey():
    detect_config_exist()
    with open("config.toml", "rb") as f:
        config = load(f)
    return config["api_key"]
    