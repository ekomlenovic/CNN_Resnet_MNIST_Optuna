import os
import joblib
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

"""
Methods to save and load an Optuna study
"""
dir = 'runs/optuna_studies'

def save_study(study, study_name, directory=dir):
    """
    Saves an Optuna study.
    
    :param study: The Optuna study object to save
    :param study_name: The name of the study
    :param directory: The directory to save the study
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    study_path = os.path.join(directory, f"{study_name}.pkl")
    joblib.dump(study, study_path)
    print(f"Study saved in {study_path}")

def load_study(study_name, directory=dir):
    """
    Loads a saved Optuna study.
    
    :param study_name: The name of the study to load
    :param directory: The directory where the study was saved
    :return: The loaded Optuna study object
    """
    study_path = os.path.join(directory, f"{study_name}.pkl")
    if not os.path.exists(study_path):
        raise FileNotFoundError(f"No study found at {study_path}")
    
    study = joblib.load(study_path)
    print(f"Study loaded from {study_path}")
    return study

def display_study_info(study):
    """
    Displays all relevant information of an Optuna study.
    
    :param study: The Optuna study object to analyze
    """
    print("\n===== Optuna Study Information =====")
    
    print("\n1. Best trial:")
    print(f"   Value: {study.best_value}")
    print("   Parameters:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")
    
    print("\n2. Study statistics:")
    print(f"   Total number of trials: {len(study.trials)}")
    print(f"   Number of completed trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
    print(f"   Number of pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
    
    print("\n3. Parameter importance:")
    importances = optuna.importance.get_param_importances(study)
    for name, importance in importances.items():
        print(f"   {name}: {importance:.4f}")
    
    print("\n4. Visualizations:")
    print("   The following visualizations will be displayed:")
    print("   - Optimization history")
    print("   - Parameter importance")
    print("   - Parallel coordinates")
    
    # Display the visualizations
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_parallel_coordinate(study)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    print("\n5. Summary of trials:")
    print(study.trials_dataframe())
