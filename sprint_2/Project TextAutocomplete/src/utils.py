from pathlib import Path
import yaml

def load_config(config_name: str):
    config_path = Path(__file__).parent.parent / "configs" / config_name
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)