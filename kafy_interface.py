# kafy_interface.py

import json
import shlex
import os
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
history = FileHistory(".kafy_history")
# Command imports


from commands.addTransformer import add_transformer
from commands.addOperation import add_operation
from commands.trainNewModel import train_new_model
# from commands.runOperation import run_operation
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
command_completer = WordCompleter([
    "AddTransformer",
    "AddOperation",
    "TrainNewModel",
    "RunOperation",
    "Exit"
], ignore_case=True)


# COMMANDS = [
#     "AddTransformer",
#     "AddOperation",
#     "TrainNewModel",
#     "RunOperation",
#     "Exit"
# ]

# def completer(text, state):
#     options = [cmd for cmd in COMMANDS if cmd.startswith(text)]
#     if state < len(options):
#         return options[state]
#     return None

# readline.set_completer(completer)
# readline.parse_and_bind("tab: complete")



# ----------------------------------------------------------
# Menu
# ----------------------------------------------------------
def print_menu():
    print("\nWelcome to KAFY Interactive CLI")
    print("Available commands:")
    print("  AddTransformer TransformerName ImplementationFile")
    print("  AddOperation OperationName TransformerName TrainingData OperationScript [Args]")
    print("  TrainNewModel OperationName TrainingData [Args]")
    print("  RunOperation OperationName InputData TrajPlugInput")
    print("  Exit")


# ----------------------------------------------------------
# Parsing user input
# ----------------------------------------------------------
# ----------------------------------------------------------
# Helper: parse args[...] block
# ----------------------------------------------------------
def parse_args_block(block):
    """
    Example:
        args[epochs=20,lr=0.0002,batch_size=4]

    Returns:
        {"epochs":20, "lr":0.0002, "batch_size":4}
    """
    if not block.startswith("args[") or not block.endswith("]"):
        raise ValueError("Malformed args[...] block")

    inner = block[len("args["):-1]

    args = {}
    for pair in inner.split(","):
        k, v = pair.split("=")

        # auto-cast numeric values
        if v.replace(".", "", 1).isdigit():
            v = float(v) if "." in v else int(v)

        args[k.strip()] = v

    return args


# ----------------------------------------------------------
# Parsing user input
# ----------------------------------------------------------
def parse_command(input_str):
    tokens = shlex.split(input_str)

    cmd = tokens[0]
    positional = []
    flags = {}

    for token in tokens[1:]:

        # special case args[...] stays intact
        if token.startswith("args[") and token.endswith("]"):
            flags["args_block"] = token

        elif token.startswith("--"):
            key, val = token[2:].split("=", 1)
            flags[key] = val

        else:
            positional.append(token)

    return cmd, positional, flags


# ----------------------------------------------------------
# Interactive loop
# ----------------------------------------------------------
def interactive_loop():

    while True:
        try:
            user_input = prompt(">> ", completer=command_completer, history=history).strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting KAFY. Goodbye!")
                break

            if not user_input:
                continue

            cmd, positional, flags = parse_command(user_input)

            # ----------------------------------------------------------
            # AddTransformer TransformerName ImplementationFile
            # ----------------------------------------------------------
            if cmd == "AddTransformer":
                if len(positional) != 2:
                    raise ValueError(
                        "Usage: AddTransformer TransformerName ImplementationFile"
                    )

                transformer_name = positional[0]
                implementation_file = positional[1]

                add_transformer(BASE_PATH, transformer_name, implementation_file)

            # ----------------------------------------------------------
            # AddOperation OperationName TransformerName TrainingData [Args] OperationScript
            # ----------------------------------------------------------
            elif cmd == "AddOperation":
                
                if len(positional) != 4:
                    raise ValueError(
                        "Usage: AddOperation OpName Transformer TrainingData Script args[k=v,...]"
                    )
                operation_name   = positional[0]
                transformer_name = positional[1]
                training_data    = positional[2]
                operation_script = positional[3]

                # ----------------------------------------------------------
                # Parse the args[...] block
                # ----------------------------------------------------------
                args_block = flags.get("args_block", None)

                if args_block is None:
                    raise ValueError("Missing args[...] block")

                args = parse_args_block(args_block)


                # ----------------------------------------------------------
                # Infer operation type from script name
                # ----------------------------------------------------------
                script_name = os.path.basename(operation_script).lower()

                if "summar" in script_name:
                    op_type = "summarization"
                elif "class" in script_name:
                    op_type = "classification"
                elif "next" in script_name:
                    op_type = "next_point"
                elif "gen" in script_name:
                    op_type = "generation"
                else:
                    raise ValueError(f"Cannot infer operation type from script name: {script_name}")

                args["operation"] = op_type
                add_operation(
                    BASE_PATH,
                    operation_name,
                    transformer_name,
                    training_data,
                    args,
                    operation_script,
                )
            # ----------------------------------------------------------
            # TrainNewModel OperationName TrainingData [Args]
            # ----------------------------------------------------------
            elif cmd == "TrainNewModel":

            #     if len(positional) < 2:
            #         raise ValueError(
            #             "Usage: TrainNewModel OperationName TrainingData"
            #         )

                if len(positional) != 2:
                    raise ValueError(
                        "Usage: TrainNewModel OperationName TrainingData args[k=v,...]"
                    )
                operation_name   = positional[0]
                training_data = positional[1]


                # ----------------------------------------------------------
                # Parse the args[...] block
                # ----------------------------------------------------------
                args_block = flags.get("args_block", None)

                if args_block is None:
                    raise ValueError("Missing args[...] block")

                args = parse_args_block(args_block)


                # ----------------------------------------------------------
                # Infer operation type from script name
                # ----------------------------------------------------------
                # script_name = os.path.basename(operation_script).lower()

                if "summar" in operation_name.lower():
                    op_type = "summarization"
                elif "class" in operation_name.lower():
                    op_type = "classification"
                elif "next" in operation_name.lower():
                    op_type = "next_point"
                elif "gen" in operation_name.lower():
                    op_type = "generation"
                else:
                    raise ValueError(f"Cannot infer operation type from script name: {operation_name.lower()}")

                args["operation"] = op_type
                train_new_model(
                    BASE_PATH,
                    operation_name,
                    training_data,
                    args,
                )


            # # ----------------------------------------------------------
            # # RunOperation OperationName InputData TrajPlugInput
            # # ----------------------------------------------------------
            # elif cmd == "RunOperation":

            #     if len(positional) < 2:
            #         raise ValueError(
            #             "Usage: RunOperation OperationName InputData --TrajPlugInput='{}'"
            #         )

            #     operation_name = positional[0]
            #     partitioning_input = positional[1]

            #     trajplug_input = json.loads(flags["TrajPlugInput"])

            #     run_operation(
            #         BASE_PATH,
            #         operation_name,
            #         partitioning_input,
            #         trajplug_input,
            #     )

            else:
                print(f"Unknown command: {cmd}")

        except Exception as e:
            print(f"[Error] {e}")


# ----------------------------------------------------------
# Main entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    print_menu()
    interactive_loop()


# AddOperation summarization T5 /export/scratch/husse408/speakingTrajectories/UseCasesForKAFY/data/summaryData.csv /export/scratch/husse408/speakingTrajectories/UseCasesForKAFY/OperationScripts/summarization.py args[epochs=20,lr=0.0002,batch_size=4]
# AddOperation summarization PEGASUS /export/scratch/husse408/speakingTrajectories/UseCasesForKAFY/data/summaryData.csv /export/scratch/husse408/speakingTrajectories/UseCasesForKAFY/OperationScripts/summarization.py args[epochs=20,lr=0.0002,batch_size=4]