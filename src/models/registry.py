from typing import Dict, Type, Any

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict: Dict[str, Type[Any]] = {}

    def register(self, name=None):
        def _register(cls):
            module_name = name if name is not None else cls.__name__
            if module_name in self._module_dict:
                 raise ValueError(f"Module {module_name} already registered in {self._name}")
            self._module_dict[module_name] = cls
            return cls
        return _register

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(f"Module {name} not found in {self._name} registry. Available: {list(self._module_dict.keys())}")
        return self._module_dict[name]

    def build(self, config: Dict[str, Any], **kwargs):
        """
        Builds a module from config/kwargs.
        Expects 'type' or 'name' in config.
        """
        # Shallow copy
        args = config.copy()
        args.update(kwargs)
        
        name = args.pop("type", None) or args.pop("name", None)
        if name is None:
            raise ValueError(f"Config must contain 'type' or 'name' key. Got: {config}")
        
        cls = self.get(name)
        return cls(**args)

BACKBONES = Registry("backbones")
QUANTIZERS = Registry("quantizers")
MODELS = Registry("models")
