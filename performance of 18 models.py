import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Create a dataframe from your data
data1 = {'ImageNet': ['vgg19', 'b0', 'b3', 'incv3', 'incresv2', 'res50',
                     'vgg19', 'b0', 'b3', 'incv3', 'incresv2', 'res50',
                     'vgg19', 'b0', 'b3', 'incv3', 'incresv2', 'res50'],
        'LanguageModel': ['xlnet', 'xlnet', 'xlnet', 'xlnet', 'xlnet', 'xlnet',
                          'gpt2', 'gpt2', 'gpt2', 'gpt2', 'gpt2', 'gpt2',
                          'bert', 'bert', 'bert', 'bert', 'bert', 'bert'],
        'MSE': [1895550501.77, 1870303297.745, 1954050566.855, 2034789256.455, 1927947048.48, 1988879540.86,
                2299886242.705, 2305111900.645, 2482492402.795, 2490070048.13, 2335512467.705, 2453295671.795,
                2223135356.065, 2356174648.16, 2599499174.67, 2530238035.04, 2355004858.365, 2354279647.645],
        'MAE': [35142.13, 34502.675, 35979.835, 37472.135, 36370.01, 36175.89,
                39827.865, 39284.615, 41062.995, 41357.52, 40371.675, 41070.245,
                38916.875, 39566.02, 41328.7, 41216.15, 39825.225, 39809.085],
        'SpearmanRho': [0.5702467561689043, 0.5667651691292284, 0.5479336983424586, 0.5182959573989351, 0.5321917543658045, 0.5257331433285833,
                        0.43554038850971283, 0.41780394509862756, 0.33130878271956804, 0.320841521038026, 0.36718367959198983, 0.3504027600690018,
                        0.4485937148428712, 0.40076351908797725, 0.31834845871146783, 0.33917647941198537, 0.39080627015675395, 0.41857346433660847]}

data2 = {
    "ImageNet": ["vgg19", "b0", "b3", "incv3", "incresv2", "res50", 
                 "vgg19", "b0", "b3", "incv3", "incresv2", "res50", 
                 "vgg19", "b0", "b3", "incv3", "incresv2", "res50"],
    "LanguageModel": ["xlnet", "xlnet", "xlnet", "xlnet", "xlnet", "xlnet", 
                      "gpt2", "gpt2", "gpt2", "gpt2", "gpt2", "gpt2", 
                      "bert", "bert", "bert", "bert", "bert", "bert"],
    "MSE": [196221382.57, 202608016.765, 196819957.37, 184643776.915, 201088429.485, 193458682.01, 
            177914669.395, 182213023.25, 172718770.655, 173348683.11, 182195662.395, 176231450.81, 
            183983294.76, 183086740.57, 178200376.19, 179863771.295, 190602553.72, 173759530.37],
    "MAE": [11325.56, 11560.985, 11353.8, 11154.725, 11335.195, 11237.24, 
            10812.625, 11152.09, 10986.875, 10961.15, 11049.965, 10774.11, 
            10825.27, 10693.26, 10726.78, 10770.325, 10844.46, 10616.72],
    "SpearmanRho": [0.47405495691527394, 0.4595366607427987, 0.4737793444836122, 0.5126468161704043, 0.4775549388734719, 0.4831680792019801, 
                    0.5246051151278782, 0.5134418360459012, 0.5610194858694934, 0.5576944423610591, 0.5353588839720993, 0.5224570614265357, 
                    0.5417970449261232, 0.5489661358781532, 0.5543958598964975, 0.5563099077476937, 0.5341978549463737, 0.573879846996175]
}


data3 = [
    {"ImageNet": "Resnet50", "LanguageModel": "BERT", "MSE": 1296020053.1175, "MAE": 26224.3975, "SpearmanRho": 0.7191541221736137},
    {"ImageNet": "B0", "LanguageModel": "XLNet", "MSE": 1288698285.8475, "MAE": 25439.0025, "SpearmanRho": 0.7087105544409652},
    {"ImageNet": "B3", "LanguageModel": "BERT", "MSE": 1326953471.7225, "MAE": 26606.7625, "SpearmanRho": 0.7030337411202983},
    {"ImageNet": "VGG19", "LanguageModel": "XLNet", "MSE": 1248373281.715, "MAE": 25082.095, "SpearmanRho": 0.7287923391142612},
    {"ImageNet": "B3", "LanguageModel": "GPT-2", "MSE": 1320212151.895, "MAE": 26323.8, "SpearmanRho": 0.7070243895442037},
    {"ImageNet": "B3", "LanguageModel": "XLNet", "MSE": 1317306942.76, "MAE": 25617.6, "SpearmanRho": 0.7072868911971437}
]



def process_and_plot(data):
    df = pd.DataFrame(data)

    # Flip MSE and MAE so higher values are better (like SpearmanRho)
    df['MSE'] = max(df['MSE']) - df['MSE']
    df['MAE'] = max(df['MAE']) - df['MAE']

    # Normalize the data so all metrics are on a scale from 0 to 1
    scaler = MinMaxScaler()
    df[['MSE', 'MAE', 'SpearmanRho']] = scaler.fit_transform(df[['MSE', 'MAE', 'SpearmanRho']])

    # Calculate the combined metrics as the sum of the normalized metrics
    df['combined_metric'] = df['MSE'] + df['MAE'] + df['SpearmanRho']

    # Create a new column "Model" which is a combination of ImageNet and LanguageModel
    df["Model"] = df["ImageNet"] + "_" + df["LanguageModel"]

    # Sort by combined metric in descending order
    df = df.sort_values(by='combined_metric', ascending=False)

    # Melt the DataFrame to format it for seaborn
    melted_df = pd.melt(df, id_vars="Model", value_vars=['MSE', 'MAE', 'SpearmanRho'])

    # Plot the data
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Model', y='value', hue='variable', data=melted_df)
    plt.title('Normalized Metrics for each model')
    plt.ylabel('Normalized Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print the ranking
    print(df[['Model', 'combined_metric']].sort_values(by='combined_metric', ascending=False).reset_index(drop=True))

# Call the function
# process_and_plot(data3)



