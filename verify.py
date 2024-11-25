import json
from langchain_community.llms import HuggingFaceHub
import sys

def verify_models():
    try:
        with open('experiments.json', 'r') as f:
            experiments = json.load(f)
    except FileNotFoundError:
        print("Error: experiments.json file not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: experiments.json is not valid JSON")
        sys.exit(1)

    failed_models = []

    for experiment in experiments:
        model_name = experiment.get('model_name')
        if not model_name:
            print(f"Warning: Missing model_name in experiment {experiment.get('name', 'unnamed')}")
            continue

        print(f"Verifying model: {model_name}")
        try:
            llm = HuggingFaceHub(
                repo_id=model_name,
                model_kwargs={"temperature": 0.1}
            )
        except Exception as e:
            print(f"Failed to load model {model_name}: {str(e)}")
            failed_models.append(model_name)
        else:
            print(f"Successfully verified {model_name}")

    if failed_models:
        print("\nThe following models failed verification:")
        for model in failed_models:
            print(f"- {model}")
        sys.exit(1)
    else:
        print("\nAll models verified successfully!")

if __name__ == "__main__":
    verify_models()
