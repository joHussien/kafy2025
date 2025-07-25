import os
import json

class OperationManager:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.LIBRARY_PATH = os.path.join(self.BASE_DIR, "operations")
        self.REGISTRY_FILE = os.path.join(self.LIBRARY_PATH, "operationRegistry.json")

    def _load_registry(self):
        if not os.path.exists(self.REGISTRY_FILE):
            return {}
        with open(self.REGISTRY_FILE, 'r') as f:
            return json.load(f)

    def _save_registry(self, registry):
        with open(self.REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=2)

    def exists(self, operation_name):
        registry = self._load_registry()
        return operation_name in registry

    def add_operation(self, operation_name, model, logic_path, rules):
        if self.exists(operation_name):
            print(f"[Info] Operation {operation_name} already defined in operations registry")
            return
        
        registry = self._load_registry()
        registry[operation_name] = {"logic_path": logic_path,
                                    "model": model,
                                    "rules": rules
        }

        self._save_registry(registry)

