# kafy.py
import json
from commands.useOperation import use_operation
# from commands.addOperation import add_operation
# from commands.trainNewModel import train_new_mode
import os
path = os.path.dirname(os.path.abspath(__file__))
def print_menu():
    print("\nWelcome to KAFY Interactive CLI")
    print("Available commands:")
    print("  - AddOperation [OperationName] --Transformer=... --OperationScript=... [--SpatialConstraints=...]")
    print("  - TrainNewModel [OperationName] [Dataset] [--TrainingArgs=...]")
    print("  - UseOperation [OperationName] --PartitioningInput=... --TrajPlugInput=...")
    print("  - Exit")

def parse_command(input_str):
    import shlex
    tokens = shlex.split(input_str)
    cmd = tokens[0]
    args = {}

    for token in tokens[1:]:
        if token.startswith("--"):
            key, val = token[2:].split("=", 1)
            args[key] = val
        else:
            args.setdefault("positional", []).append(token)

    return cmd, args

def interactive_loop():
    while True:
        try:
            user_input = input("\n>> ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting KAFY. Goodbye!")
                break
            elif not user_input:
                continue

            command, args = parse_command(user_input)

            if command == "AddOperation":
                print("AddOperation called")  # Replace with actual function
                # add_operation(args["positional"][0], args["Transformer"], args["OperationScript"], args.get("SpatialConstraints"))
            elif command == "TrainNewModel":
                print("TrainNewModel called")  # Replace with actual function
                # train_new_model(args["positional"][0], args["positional"][1], args.get("TrainingArgs", ""))
            elif command == "UseOperation":
                trajplug_input = json.loads(args["TrajPlugInput"])
                use_operation(path, args["positional"][0], args["PartitioningInput"], trajplug_input)
            else:
                print(f"Unknown command: {command}")
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    print_menu()
    
    interactive_loop()


# USAGE Commands Templates

# UseOperation generation --PartitioningInput=trajectories.geojson --TrajPlugInput='{"MaxLength": 20, "Count": 3, "MaxTrials": 50}'
# UseOperation nextPointPredict  --PartitioningInput=trajectories.geojson  --TrajPlugInput='{"NumOfPoints": 5}'
# UseOperation classify  --PartitioningInput=trajectories.geojson  --TrajPlugInput='{"Labels": ["car", "bike", "bus"]}'
# UseOperation summarization  --PartitioningInput=trajectories.geojson  --TrajPlugInput='{"Method": "Douglas-Peucker"}'
# UseOperation imputation --PartitioningInput=trajectories.geojson