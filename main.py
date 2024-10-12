import argparse
from utils.methods import *

def main():
    parser = argparse.ArgumentParser(description="Run different models. Default is CNN")
    parser.add_argument('-cnn', action='store_true', help="Run CNN model")
    parser.add_argument('-resnet', action='store_true', help="Run ResNet model")
    parser.add_argument('-study-cnn', action='store_true', help="Load study and display information about CNN study (need to be run after CNN model)")
    parser.add_argument('-study-resnet', action='store_true', help="Load study and display information about ResNet study (need to be run after ResNet model)")
    args = parser.parse_args()

    if args.study_cnn:
        study = load_study("study_cnn")
        display_study_info(study)
    elif args.study_resnet:
        study = load_study("study_resnet")
        display_study_info(study)
    elif args.resnet:
        run_optuna_with_asha_and_tpe("resnet")
    else:
        run_optuna_with_asha_and_tpe("cnn")

    


if __name__ == "__main__":
    main()
