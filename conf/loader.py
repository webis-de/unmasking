from conf.interfaces import ConfigLoader

from typing import Any, Dict
import yaml


class YamlLoader(ConfigLoader):
    """
    Parser for YAML config files.
    """

    def __init__(self):
        self._config = {}

    def get(self, name: str = None) -> Any:
        """
        Get configuration option. Use dot notation to reference hierarchies of options.

        :param name: dot-separated config option path, None to get full config dict
        :return: option value
        :raise: KeyError if option not found
        """
        if name is None:
            return self._config

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

    def set(self, cfg: Dict[str, Any]):
        self._config = cfg

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
            if keys[0] not in parsed_cfg:
                parsed_cfg[keys[0]] = {}
            parsed_cfg[keys[0]].update(self._parse_dot_notation({keys[1]: cfg[i]}))
        return parsed_cfg

    def save(self, file_name: str) -> Any:
        with open(file_name + ".yml", "w") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)


class JobConfigLoader(YamlLoader):
    """
    Job configuration loader class with fallback to default configuration.
    """

    _default_config = None

    def __init__(self, cfg: Dict[str, Any] = None):
        """
        :param cfg: optional configuration dict to construct configuration from
        """
        super().__init__()

        if self._default_config is None:
            self._default_config = YamlLoader()
            self._default_config.load("etc/defaults.yml")

        if cfg is not None:
            self.set(cfg)

    def load(self, filename: str):
        super().load(filename)
        self._config.update(self._resolve_inheritance(self._config))

    def set(self, cfg : Dict[str, Any]):
        super().set(cfg)
        self._config.update(self._resolve_inheritance(self._config))
    
    def _resolve_inheritance(self, d: Dict[str, Any], path: str = ""):
        for k in d:
            if k.endswith("%"):
                t = type(d[k])
                p = path + "." + k[0:-1] if path != "" else k[0:-1]
                
                if t is not dict and t is not list:
                    raise KeyError("Config option '{}' is of non-inheritable type {}".format(p, t))
                
                try:
                    inherit = self._default_config.get(p)
                except KeyError:
                    raise KeyError("Config option '{}' has no inheritable defaults".format(p))
                
                d[k[0:-1]] = inherit
                if t is dict:
                    d[k[0:-1]].update(self._resolve_inheritance(d[k], p))
                elif t is list:
                    d[k[0:-1]].extend(d[k])
                
                del d[k]
            elif type(d[k]) is dict:
                d[k] = self._resolve_inheritance(d[k], path + "." + k if path != "" else k)

        return d
