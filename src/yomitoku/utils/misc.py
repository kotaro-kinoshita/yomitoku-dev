from omegaconf import OmegaConf


def load_config(config_path):
    with open(config_path, "r") as file:
        config = OmegaConf.load(file)
    return config


def load_charset(charset_path):
    with open(charset_path, "r") as f:
        charset = f.read()
    return charset
