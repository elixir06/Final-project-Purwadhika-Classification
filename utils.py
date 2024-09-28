import matplotlib.pyplot as plt
import numpy as np

def filter_model(old_dict: dict, filter: list):
    """
    This function is created in order filter the dictionary containing the model
    and important variable value for plotting roc curves.
    Parameters: old_dict: dictionary, containing the fpr, treshold, and auc value
                            of different models
                filter : list, list of model names to be filtered
    
    Returns: new_dict, new dictionary containing only selected model
                       and it's variables values
    """

    new_dict = {key : old_dict[key] for key in filter}
    return new_dict

def mean_value_iterations(data: dict):
    """
    This function is created in order get the new dictionary containing the model
    with mean value of each variable in used (tpr, treshold, auc).
    Parameters: data: dictionary, containing the fpr, treshold, and auc value
                            of different models with different iterations
    
    Returns: new_data, dictionary, new dictionary containing model
                       and the mean value of each variables
    """
    new_data = {}

    for model in data.keys():
        for var in data[model].keys():
            if var != "y_pred_proba":
                result = (data[model][var][0] + data[model][var][1] + data[model][var][2] + data[model][var][3] + data[model][var][4])/5
                if not model in new_data.keys():
                    new_data[model] = {var : result}
                else:
                    new_data[model][var] = result
    return new_data


###### ROC PLOT #####################################

def plot_roc_iterations(model_dict, model_name):
    """
    This function is created in order to plot the interpolated roc curve for a specific model
    Parameters: model_dict: dictionary, containing the fpr, treshold, and auc value
                            of different models with different iterations
                model_name : str, name of the model for which the curve is to be plotted
    
    Returns: None, it just plots the interpolated roc curve
    """

    plt.figure(figsize=(8, 6))
    
    # Access the model's data
    model_data = model_dict[model_name]
    
    # FPR values ranging from 0 to 1 with a 0.01 step
    fpr = np.linspace(0, 1, 101)

    # intialize anotation as false to make anotation
    anotated = False
    
    # Iterate through each iteration
    for iteration in model_data['tpr']:
        tpr = model_data['tpr'][iteration]
        thresholds = model_data['threshold'][iteration]
        roc_auc = model_data['auc'][iteration]
        
        plt.plot(fpr, tpr, lw=2, label=f'Iteration {iteration} (AUC = {roc_auc:.3f})')
        
        if not anotated:
            # Annotate the interpolated thresholds at every 0.1 FPR
            for i in range(len(fpr)):
                if np.isclose(fpr[i], np.arange(0.1, 1.1, 0.1), atol=0.001).any():
                    plt.annotate(f'{thresholds[i]:.2f}', (fpr[i], tpr[i]), 
                                textcoords="offset points", xytext=(-10,10), ha='center')

            anotated = True
    # Diagonal line for no-skill classifier
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()


def plot_roc_standard(tpr, thresholds, roc_auc):

    # initialize the fpr
    fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for no-skill classifier
    
    
    # Annotate the interpolated thresholds at every 0.1 FPR
    for i in range(len(fpr)):
        if np.isclose(fpr[i], np.arange(0.1, 1.1, 0.1), atol=0.001).any():  # Check if mean_fpr is close to 0.1, 0.2, ..., 1.0
            plt.annotate(f'{thresholds[i]:.2f}', (fpr[i], tpr[i]), 
                        textcoords="offset points", xytext=(-10,10), ha='center')


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

###### Best Treshold Model and plotting #################

def calculate_target_slope(p, n, c_fp, c_fn):
    """
    Calculate the target slope based on the given formula
    """
    return (n / p) * (c_fp / c_fn)

def find_optimal_threshold(tpr, threshold, target_slope):
    """
    Find the optimal threshold based on the ROC curve and cost ratio,
    using constant interval gradients
    """
    fpr = np.linspace(0, 1, 101)
    # calculate the gradient for each tpr and fpr
    gradients = np.diff(tpr) / np.diff(fpr)

    # Find the index where gradient is closest to the target slope
    optimal_idx = np.argmin(np.abs(gradients - target_slope))
    
    # Interpolate to find the corresponding threshold
    optimal_threshold= threshold[optimal_idx + 1]
    optimal_fpr = fpr[optimal_idx+1]
    
    return optimal_fpr, optimal_threshold, gradients


def plot_roc_curve_with_optimal_threshold(tpr, thresholds, optimal_threshold, auc):
    """
    Plot ROC curve with optimal threshold point and constant interval points
    """
    fpr = np.linspace(0, 1, 101)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Find and plot the optimal threshold point
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
    
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10, label=f'Optimal threshold ({optimal_threshold:.2f})')

    # Annotate the optimal threshold point with FPR and TPR values
    plt.annotate(f'({optimal_fpr:.3f}, {optimal_tpr:.3f})',
                 xy=(optimal_fpr, optimal_tpr),
                 xytext=(optimal_fpr + 0.05, optimal_tpr - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, color='black')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Constant Interval Gradients')
    plt.legend(loc="lower right")
    plt.show()


def calculate_revenue(tp, fp, p, n):
    """
    This function calculates and compares the net revenue generated by
    a machine learning (ML) model against the revenue that would have been
    generated without the model. The function considers the true positives (tp),
    false positives (fp), total positives (p), and total negatives (n) in the dataset
    to determine the revenue impact of using the ML model.
    
    The net revenue is calculated based on the income generated from successful
    predictions (true positives) and the losses incurred from incorrect predictions (false positives),
    factoring in the associated costs of each call. The function then prints out
    the net revenue with and without the ML model and the difference in revenue
    resulting from the use of the ML model.
    """
    call_cost = 7
    tp_income = 1000 - call_cost
    fp_loss = 120 + call_cost

    # with ml
    actual_revenue_ml = tp*tp_income - fp*fp_loss
    actual_revenue_non_ml = p*tp_income - n*fp_loss
    print("Net Revenue with ML : €", np.round(actual_revenue_ml))
    print("Net Revenue without ML : €", np.round(actual_revenue_non_ml))
    print("Revenue Difference by Using ML : €", np.round(actual_revenue_ml - actual_revenue_non_ml))
