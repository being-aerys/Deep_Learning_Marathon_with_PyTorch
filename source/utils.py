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

import torch
from timeit import default_timer

def print_training_time(start: float, end: float, device: torch.device = None) -> float:
    """
    Prints and returns the difference between the end time and the start time.  
    """
    total_time_taken = end - start
    print(f"Epoch training time: {round(total_time_taken,3)} seconds.")
    return total_time_taken

# method taken from Daniel Bourke
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


def training_step(model: torch.nn.Module,
                  data_loader : torch.utils.data.DataLoader,
                  loss_function : torch.nn.Module,
                  optimizer : torch.optim,
                  accuracy_func,
                  device : torch.device):
    """
    Trains a passed torch.nn.Module object using the passed DataLoader object.
    """
    cumulative_tr_loss, cumulative_tr_acc = 0, 0
    
    model.train()
    for batch_num, (X, y) in enumerate(data_loader):
        
        X, y = X.to(device=device), y.to(device=device)

        y_pred_batch = model(X)

        batch_loss = loss_function(y_pred_batch, y)
        tr_acc = accuracy_func(y, y_pred_batch.argmax(dim = 1))
        cumulative_tr_acc += tr_acc
        
        # accumulate training loss for an epoch for plotting purpose
        # ensure to detach the variable value from the computation graph
        cumulative_tr_loss += batch_loss.to("cpu").item()

        # wipe out garbage gradients accumulated
        optimizer.zero_grad()

        # backpropagation to calculate gradients
        batch_loss.backward()

        # optimizer the model parameters once per batch
        optimizer.step()
      
    
    # average training loss per batch
    avg_tr_loss_per_batch = cumulative_tr_loss/len(data_loader)
    avg_tr_acc_per_batch = cumulative_tr_acc/len(data_loader)

    print("Avg tr loss/batch: ", round(avg_tr_loss_per_batch,3), ", Avg tr acc/batch: ", round(avg_tr_acc_per_batch,3))
    return avg_tr_acc_per_batch, avg_tr_loss_per_batch


def test_step(model: torch.nn.Module,
                data_loader : torch.utils.data.DataLoader,
                loss_function : torch.nn.Module,
                accuracy_func,
                device : torch.device):
    """
    Performs a testing loop using the passed torch.nn.Module object over the passed DataLoader object.
    """

    cumulative_test_loss, cumulative_test_acc = 0, 0

    model.eval()
    with torch.inference_mode():

        for batch_num, (X,y) in enumerate(data_loader):

            X, y = X.to(device=device), y.to(device=device)

            test_preds = model(X)
            test_loss = loss_function(test_preds, y)
            cumulative_test_loss += test_loss.item()

            test_acc = accuracy_func(y, test_preds.argmax(dim = 1))
            cumulative_test_acc += test_acc
        
        # calculate average test loss and test accuracy per batch
        avg_test_loss_per_batch = cumulative_test_loss/len(data_loader)
        avg_test_acc_per_batch = cumulative_test_acc/len(data_loader)
        
        print("Avg test loss/batch:", round(avg_test_loss_per_batch,3),", Average test acc/batch:", round(avg_test_acc_per_batch,3))
    
    return avg_test_acc_per_batch, avg_test_loss_per_batch