import os
import argparse
import pandas as pd
import json
from rag_evaluate import run_evaluation as run_rag_evaluation
from fine_tune import run_fine_tuning
from ft_evaluate import run_evaluation as run_ft_evaluation

def load_experiments(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def run_experiments(experiments):
    results = []
    
    for exp in experiments:
        try:
            print(f"\nRunning experiment: {exp['name']}")
            
            print("\nRunning RAG evaluation...")
            rag_output = run_rag_evaluation(
                exp['model_name'],
                exp['test_file'],
                exp['doc_dirs'],
                exp['is_multiple_choice']
            )
            
            print("\nRunning fine-tuning...")
            run_fine_tuning(
                exp['model_name'],
                exp['doc_dirs']
            )
            
            print("\nEvaluating fine-tuned model...")
            ft_output = run_ft_evaluation(
                exp['model_name'],
                exp['test_file'],
                exp['is_multiple_choice']
            )
            
            results.append({
                'Experiment': exp['name'],
                'Model': exp['model_name'],
                'Documents': ', '.join(exp['doc_dirs']),
                'Test Set': exp['test_file'],
                'Multiple Choice': exp['is_multiple_choice'],
                'RAG Score': rag_output if rag_output else None,
                'Fine-tuned Score': ft_output if ft_output else None
            })
        except Exception as e:
            print("Experiment Failed")
            print(e)
        
    results_df = pd.DataFrame(results)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'experiment_results_{timestamp}.csv', index=False)
    print(f"\nResults saved to experiment_results_{timestamp}.csv")
    print("\nResults summary:")
    print(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments from a JSON configuration file")
    parser.add_argument("config_file", help="Path to the JSON configuration file")
    args = parser.parse_args()

    experiments = load_experiments(args.config_file)
    
    run_experiments(experiments)
