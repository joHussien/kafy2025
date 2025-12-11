# commands/addTransformer.py

import os
from coreComponents.transformers_plugin import TransformersPlugin

def add_transformer(base_path, transformer_name, implementation_file):
    impl_path = os.path.abspath(implementation_file)

    if not os.path.exists(impl_path):
        raise FileNotFoundError(f"Transformer implementation file not found: {impl_path}")

    plugin = TransformersPlugin(base_path)
    plugin.add_transformer(transformer_name, impl_path)