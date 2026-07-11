import itertools
import subprocess
import time

def get_all_combinations(feature_list):
    """
    Generates all possible combinations of the provided features.
    (Solo, pairs, and all combined).
    """
    all_combinations = []
    # Loop from length 1 to length 3 (solo, pairs, all)
    for r in range(1, len(feature_list) + 1):
        combinations_object = itertools.combinations(feature_list, r)
        combinations_list = list(combinations_object)
        all_combinations.extend(combinations_list)
        
    return all_combinations

def main():
    # The exact names of the features we extracted
    available_features = ['stats', 'lbp', 'hog']
    models_to_test = ['rf', 'svm'] # Test both Random Forest and SVM
    
    combinations = get_all_combinations(available_features)
    
    print(f"Found {len(combinations)} feature combinations to test.")
    print("Starting the ultimate ablation study...\n" + "-"*40)
    
    total_runs = len(combinations) * len(models_to_test)
    current_run = 1

    for model in models_to_test:
        for combo in combinations:
            # Convert tuple ('stats', 'hog') to a space-separated string "stats hog"
            features_str = " ".join(combo)
            combo_name = "_".join(combo)
            
            print(f"\n[{current_run}/{total_runs}] Running {model.upper()} with features: {combo_name}")
            
            # The command we would normally type in the terminal
            command = f"python src/train_ml.py --model {model} --features {features_str}"
            
            # Run the command
            start_time = time.time()
            subprocess.run(command, shell=True)
            end_time = time.time()
            
            print(f"Finished {model.upper()} with {combo_name} in {end_time - start_time:.1f} seconds.")
            current_run += 1

    print("\n" + "="*40)
    print("All experiments completed! Check MLflow for the results.")

if __name__ == "__main__":
    main()