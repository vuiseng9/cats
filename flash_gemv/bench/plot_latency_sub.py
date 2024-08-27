import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams['font.family'] = 'serif'  # Use 'sans-serif' for Arial/Helvetica
plt.rcParams['font.serif'] = ['Liberation Sans']  # Change to your preferred serif font
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 16   
plt.rcParams['axes.titlesize'] = 16   
plt.rcParams['axes.labelsize'] = 16  
plt.rcParams['xtick.labelsize'] = 12   
plt.rcParams['ytick.labelsize'] = 12   
plt.rcParams['legend.fontsize'] = 16   

cmap_name = 'tab10'
BLUE, ORANGE, GREEN, RED, PURPLE = sns.color_palette(cmap_name)[:5]
sns.color_palette(cmap_name)

# Path to your CSV file
# file_paths = ["final_methods_mistral7B_fp32.csv", "final_methods_llama7B_fp32.csv"]
# titles = ["CATS of Mistral-7B’s MLP", "CATS of Llama2-7B’s MLP"]
# models = ["Mistral", "Llama2"]

file_paths = ["final_methods_mistral7B_fp32.csv"]
titles = ["CATS of Mistral-7B’s MLP"]
models = ["Mistral"]

for idx in range(len(file_paths)):
    file_path = file_paths[idx]
    # Lists to store the extracted values
    sparsity_list = []
    dense_latency_list = []
    baseline_latency_list = []
    ours_latency_list = []
    optimal_latency_list = []
    scap_latency_list = []

    # Open the file and read data
    with open(file_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)

        # Extract data from each row
        for row in csv_reader:
            if row:  # Check if the row is not empty
                print(row[0])
                if row[0] in sparsity_list:
                    continue
                else:
                    sparsity_list.append(float(row[0]))
                    dense_latency_list.append(float(row[2]))
                    baseline_latency_list.append(float(row[3]))
                    ours_latency_list.append(float(row[-3]))
                    optimal_latency_list.append(float(row[-2]))
                    scap_latency_list.append(float(row[-1]))

    print(ours_latency_list, baseline_latency_list)
    # Creating a DataFrame from the lists
    data = {
        "Sparsity": sparsity_list,
        "Dense Latency": dense_latency_list,
        "CATS Latency": ours_latency_list,
        "Baseline Latency": baseline_latency_list,
        "Optimal Latency": optimal_latency_list,
        "SCAP Latency": scap_latency_list
    }
    df = pd.DataFrame(data)

    # Set the line type as scatter plot+line plot
    sns.set_theme(context='paper', style='white')

    # Create the first figure
    plt.figure(figsize=(4, 4))
    plt.figure(1 + idx * 2)
    sns.lineplot(data=df, x="Sparsity", y="Dense Latency", color=GREEN, marker="o", errorbar=None)
    sns.lineplot(data=df, x="Sparsity", y="CATS Latency", color=BLUE, marker="o", errorbar=None)
    # sns.lineplot(data=df, x="Sparsity", y="Optimal Latency", marker="o", errorbar=None)
    sns.lineplot(data=df, x="Sparsity", y="SCAP Latency", color=ORANGE, marker="o", errorbar=None)
    plt.ylabel("Latency (ms)")
    plt.xlabel("FFN Sparsity (%)")
    plt.legend(["Dense", "CATS", "SCAP"])
    # plt.tight_layout()
    plt.grid(True, linestyle='dotted')
    plt.title("Latency of Mistral-7B's FFN (SwiGLU)")
    save_dir = os.path.join(os.getenv("CATS_RESPATH", ""), "speedup", "figures")
    plot_name = f"mistral_ffn_sparsity_latency_sweep"
    print(f"Saving {plot_name}.png/pdf to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + f"/{plot_name}.png")
    plt.savefig(save_dir + f"/{plot_name}.pdf")

    exit()
    # Create the second figure with the additional line
    plt.figure(2 + idx * 2)
    sns.lineplot(data=df, x="Sparsity", y="Dense Latency", marker="o", errorbar=None)
    sns.lineplot(data=df, x="Sparsity", y="CATS Latency", marker="o", errorbar=None)
    sns.lineplot(data=df, x="Sparsity", y="Optimal Latency", marker="o", errorbar=None)
    sns.lineplot(data=df, x="Sparsity", y="Baseline Latency", marker="o", errorbar=None)
    sns.lineplot(data=df, x="Sparsity", y="SCAP Latency", marker="o", errorbar=None)
    plt.ylabel("Latency(ms)")
    plt.xlabel("Sparsity")
    plt.legend(["Dense", "CATS", "Optimal (Cropped Weight)", "Baseline", "SCAP"])
    plt.title(titles[idx])
    print(f"Saving figure 6 to {save_dir}")
    plt.savefig(save_dir + f"/fig6: {models[idx]}_mlp_ablation_latency.png")
