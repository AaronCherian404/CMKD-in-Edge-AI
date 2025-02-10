import matplotlib.pyplot as plt
import pandas as pd

def plot_training_progress(log_file):
    df = pd.read_csv(log_file)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.savefig('training_progress.png')
    plt.close()
