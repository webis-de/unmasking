from conf.interfaces import ConfigLoader

from typing import Any, Dict
import yaml


class YamlLoader(ConfigLoader):
    """
    Parser for YAML config files.
    """

    def __init__(self):
        self._config = {}

    def get(self, name: str) -> Any:
        """
        Get configuration option. Use dot notation to reference hierarchies of options.

        :param name: dot-separated config option path
        :return: option value
        :raise: KeyError if option not found
        """
        keys = name.split(".")
        cfg = self._config
        for k in keys:
            if k not in cfg:
                raise KeyError("Missing config option '{}'".format(name))

            cfg = cfg[k]
        return cfg

    def load(self, filename: str):
        with open(filename, 'r') as f:
            cfg = yaml.safe_load(f)

        if type(cfg) is not dict:
            raise RuntimeError("Invalid configuration file")

        self._config = self._parse_dot_notation(cfg)

    def _parse_dot_notation(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        parsed_cfg = {}
        for i in cfg:
            if "." not in i:
                if type(cfg[i]) is not dict:
                    parsed_cfg[i] = cfg[i]
                else:
                    parsed_cfg[i] = self._parse_dot_notation(cfg[i])
                continue

            keys = i.split(".", 1)
            parsed_cfg[keys[0]] = self._parse_dot_notation({keys[1]: cfg[i]})
        return parsed_cfg


class JobConfigLoader(YamlLoader):
    """
    Job configuration loader class with fallback to default configuration.
    """

    def __init__(self):
        super().__init__()
        self._default_config = YamlLoader()
        self._default_config.load("etc/defaults.yml")

    def get(self, name: str) -> Any:
        try:
            return super().get(name)
        except KeyError:
            return self._default_config.get(name)
