import matplotlib.pyplot as plt
def plot_predictions(train_data, train_labels, test_data, test_labels, test_predictions = None):
    """
    plots training and test data.
    """

    plt.figure(figsize=(10,7))

    #plot training data in green
    plt.scatter(train_data, train_labels, c = "g", marker = "8", s=30, label = "training data")

    #plot testing data in blue
    plt.scatter(test_data, test_labels, c = "b", marker = "8", s=30, label = "test data")

    # plot predictions if present
    if test_predictions is None:
        pass
    else:
        plt.scatter(test_data, test_predictions, c = "r", marker = "8", s = 30, label = "test predictions")

    # show legends
    plt.legend(prop = {"size" : 8})

