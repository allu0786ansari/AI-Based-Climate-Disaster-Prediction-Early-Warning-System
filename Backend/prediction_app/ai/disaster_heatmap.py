import matplotlib.pyplot as plt
import pandas as pd

def generate_heatmap():
    data = pd.read_csv("../dataset/disaster_data.csv")
    plt.scatter(data['longitude'], data['latitude'], c=data['intensity'], cmap='hot', alpha=0.7)
    plt.colorbar()
    plt.savefig("../static/heatmap.png")
