from LRUandJaxLIB import *
from alive_progress import alive_bar


# Define the model

Encoding_layer = init_mlp_parameters([1,3,5,10])
LRU_sub_1 = init_lru_parameters(10, 10)
LRU_nonlinear_part = init_mlp_parameters([10,10,10])
Decoding_layer = init_mlp_parameters([10,5,3,1])

# Define the hyperparameters and model
model_and_hyperparameters = {"Model Parameters" : (Encoding_layer, LRU_sub_1, LRU_nonlinear_part, Decoding_layer),
                             "Learning Rate" : 0.001,
                             "batch_size " : 10,
                             "epochs" : 10,
                             "optimizer" : "AdamW",
                             "loss_function" : "CrossEntropy",
                             "metric" : "Accuracy",
                             "training dataset circumstanct" : "Steady frequeny, CN0 15 dbHz"
                             } 

# Load the data
#
def train_model(model_and_hyper_parameters_for_function, data_file_path):
    """
    Train the model with the given hyperparameters and data

    model_and_hyper_parameters : dictionary
    data_file_path : string

    return : None
    """
    # Load the data
    train_sequences, train_labels, test_sequences, test_labels = load_data(data_file_path, model_and_hyper_parameters_for_function["batch_size "], 1)

    # Train the model
    model_parameters = model_and_hyper_parameters_for_function["Model Parameters"]

    # Setup the optimizer
    if model_and_hyper_parameters_for_function["optimizer"] == "AdamW":
        optimizer = optax.adamw(learning_rate=model_and_hyper_parameters_for_function["Learning Rate"])

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    epoch_model_parameters = [model_and_hyper_parameters_for_function["Model Parameters"]]

    # Train the model
    for epoch in range(model_and_hyper_parameters_for_function["epochs"]):
        with alive_bar(len(train_sequences)) as bar:
            for i in range(len(train_sequences)):
                grads = model_grad(train_sequences[i], train_labels[i], model_parameters)
                updates, optimizer_state = optimizer.update(grads, optimizer_state)
                model_parameters = optax.apply_updates(model_parameters, updates)
                bar()

        # Calculate the loss and accuracy
        train_acc.append(np.mean([accuracy(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(train_sequences, train_labels)]))
        test_acc.append(np.mean([accuracy(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(test_sequences, test_labels)]))
        train_loss.append(np.mean([loss_fn(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(train_sequences, train_labels)]))
        test_loss.append(np.mean([loss_fn(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(test_sequences, test_labels)]))

        epoch_model_parameters.append(model_parameters)

    # Save the model
    model_and_hyper_parameters_for_function["Accuracy Measurements"] = {"Training accuracy" : train_acc,
                                                                        "Testing accuracy" : test_acc}
    model_and_hyper_parameters_for_function["Loss Measurements"] = {"Training loss" : train_loss,
                                                                    "Testing loss" : test_loss}

    model_and_hyper_parameters_for_function["Model Parameters"] = epoch_model_parameters

    return None