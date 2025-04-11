import os
import glob
import pandas as pd

folder = "/home/joao/Documentos/repos/snhacks/ml_for_ftir/000_principal_wavenumber"

principal_wavenumber_results_path = "principal_wavenumber_results.csv"

def create_single_principal_wavenumber_csv(csv_folder, final_csv_path):
    csv_list = glob.glob(os.path.join(csv_folder, "**/**.csv"))
    all_dataframes = []
    for csv_file in csv_list:
        filename = os.path.basename(csv_file)
        target_to_predict = os.path.basename(os.path.dirname(csv_file))
        filename=filename.split("wavenumbers_importance_")[-1].replace(".csv", "")
        sample_type, train_percentage, test_name, _, accuracy=filename.split("_")
        train_percentage=float(train_percentage.split("pct")[0])/100
        accuracy=float(accuracy)

        model = " ".join(test_name.split(" ")[:-1])
        search_type = test_name.split(" ")[-1].replace("(", "").replace(")", "")


        df = pd.read_csv(csv_file)
        df["sample_type"] = sample_type
        df["train_percentage"] = train_percentage
        df["test_name"] = test_name
        df["target_variable"] = target_to_predict
        df["accuracy"] = accuracy
        df["model"] = model
        df["search_type"] = search_type


        all_dataframes.append(df)
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(final_csv_path, index=False)
    return

create_single_principal_wavenumber_csv(folder, principal_wavenumber_results_path)