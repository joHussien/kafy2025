import os
import json
import importlib.util
import importlib
import inspect

class SpatialConstraintsPlugin:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.LIBRARY_PATH = os.path.join(self.BASE_DIR, "constraintsLibrary")
        self.RULES_FILE = os.path.join(self.LIBRARY_PATH, "fullRules.py")
        print(self.RULES_FILE )
        self.REGISTRY_FILE = os.path.join(self.LIBRARY_PATH, "registry.json")
        self.OPERATIONS_DIR = os.path.join(self.BASE_DIR, "operations")

    def _load_registry(self):
        if not os.path.exists(self.REGISTRY_FILE):
            return {}
        with open(self.REGISTRY_FILE, 'r') as f:
            return json.load(f)

    def _save_registry(self, registry):
        with open(self.REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=2)

    def exists(self, function_name):
        registry = self._load_registry()
        return function_name in registry

    def add_rule(self, function_name, function_path):
        registry = self._load_registry()
        if function_name in registry:
            print(f"[Info] Rule {function_name} already exists")
        else:
            logic = self.load_functions_from_file(function_path)
            if not hasattr(logic, 'apply_constraint'):
                raise AttributeError("Constraint script must contain a function named 'apply_constraint'")
            
            registry[function_name] = function_path
            self._save_registry(registry)

    def register(self, function_name, operation_name):
        op_constraints_path = os.path.join(self.BASE_DIR, "operations", operation_name, "constraints.txt")
        os.makedirs(os.path.dirname(op_constraints_path), exist_ok=True)

        if os.path.exists(op_constraints_path):
            with open(op_constraints_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
        else:
            lines = []

        if function_name not in lines:
            with open(op_constraints_path, 'a') as f:
                f.write(function_name + "\n")

    def load_functions_from_file(self, file_path):
        spec = importlib.util.spec_from_file_location("custom_constraints", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return inspect.getmembers(module, inspect.isfunction)

    def process_constraints_file(self, file_path, operation_name):
        print(f"[Info] Loading constraints from: {file_path}")
        functions = self.load_functions_from_file(file_path)

        rules = []
        for name, func in functions:
            source = inspect.getsource(func)
            if not self.exists(name):
                self.add_rule(name, source)
            rules.append(name)

        print(f"[Info] Constraints registered for operation: {operation_name}")
        return rules

    def load(self, operation_name):
        op_constraints_path = os.path.join(self.BASE_DIR, "operations", operation_name, "constraints.txt")
        registry = self._load_registry()
        print(registry)

        if not os.path.exists(op_constraints_path):
            return []

        with open(op_constraints_path, 'r') as f:
            function_names = [line.strip() for line in f.readlines() if line.strip()]
        print(function_names)
        loaded_functions = []
        for fn in function_names:
            module_path = registry.get(fn)
            if not module_path:
                raise ImportError(f"Function {fn} not found in registry")

            module_name, func_name = module_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            loaded_func = getattr(module, func_name)
            loaded_functions.append(loaded_func)

        return loaded_functions