import pandas as pd
import matplotlib.pyplot as plt

# Save history for visualization
def save_history(history, filename='history.csv'):
    df = pd.DataFrame(history.history)
    df.to_csv(filename, index=False)
    return df

def visualize_history(history):
    # Extract the Dice coefficient values from the history object
    val_dice = history.history.get('val_dice_coefficient', [])

    # Plot the Dice coefficient across epochs
    plt.plot(val_dice, label='Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient During Training')
    plt.legend()
    plt.grid(True)
    plt.show()
