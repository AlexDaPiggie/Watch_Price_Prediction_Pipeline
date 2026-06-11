from Data_Cleaning import main as run_data_cleaning
from Modeling import run_modeling


def main():
    run_data_cleaning()
    return run_modeling()


if __name__ == "__main__":
    main()
