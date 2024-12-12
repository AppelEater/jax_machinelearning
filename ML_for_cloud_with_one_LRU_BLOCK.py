from LRUandJaxLIB import *
from alive_progress import alive_bar
from datetime import datetime
import optax


# Make a train model function
#
def train_model(model_and_hyper_parameters_for_function, data_file_path):
    """
    Train the model with the given hyperparameters and data

    model_and_hyper_parameters : dictionary
    data_file_path : string

    return : None
    """
    # Load the data
    train_sequences, train_labels, test_sequences, test_labels = load_data(data_file_path, model_and_hyper_parameters_for_function["batch_size "], 9)

    # Train the model
    model_parameters = model_and_hyper_parameters_for_function["Model Parameters"]

    # Setup the optimizer
    if model_and_hyper_parameters_for_function["optimizer"] == "AdamW":
        optimizer = optax.adamw(learning_rate=model_and_hyper_parameters_for_function["Learning Rate"], weight_decay=0.05)
    elif model_and_hyper_parameters_for_function["optimizer"] == "Adam":
        optimizer = optax.adam(learning_rate=model_and_hyper_parameters_for_function["Learning Rate"])

    opt_state = optimizer.init(model_parameters)


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
                updates, opt_state = optimizer.update(grads, opt_state)
                model_parameters = optax.apply_updates(model_parameters, updates)
                bar()

        # Calculate the loss and accuracy
        train_acc.append(np.mean([accuracy(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(train_sequences, train_labels)]))
        test_acc.append(np.mean([accuracy(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(test_sequences, test_labels)]))
        train_loss.append(np.mean([loss_fn(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(train_sequences, train_labels)]))
        test_loss.append(np.mean([loss_fn(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(test_sequences, test_labels)]))
        print(f"Test acc {test_acc}")
        epoch_model_parameters.append(model_parameters)

    # Save the model
    model_and_hyper_parameters_for_function["Accuracy Measurements"] = {"Training accuracy" : train_acc,
                                                                        "Testing accuracy" : test_acc}
    model_and_hyper_parameters_for_function["Loss Measurements"] = {"Training loss" : train_loss,
                                                                    "Testing loss" : test_loss}

    model_and_hyper_parameters_for_function["Model Parameters"] = epoch_model_parameters

    return model_and_hyper_parameters_for_function


# Define the model
Encoding_layer = init_mlp_parameters([1,3,5,10])
LRU_sub_1 = init_lru_parameters(256, 10, r_min =0.9, r_max=0.999)
LRU_nonlinear_part = init_mlp_parameters([10,10,10])
Decoding_layer = init_mlp_parameters([10,10,9])

# Learning rate
learning_rates = jnp.logspace(-3.3,-4.5,10)
batch_sizes = np.arange(13,18,1)

# Define the hyperparameters and model
for idx, learning_rate in enumerate(learning_rates):
    model_and_hyperparameters = {"Model Parameters" : (Encoding_layer, LRU_sub_1, LRU_nonlinear_part, Decoding_layer),
                                "Learning Rate" : learning_rate,
                                "batch_size " : 15,
                                "epochs" : 20,
                                "optimizer" : "Adam",
                                "loss_function" : "CrossEntropy",
                                "metric" : "Accuracy",
                                "training dataset circumstanct" : "Steady frequeny, CN0 15 dbHz"
                                } 

    with open(f"grid_search5/results{idx} time {datetime.now()}.pkl", "wb") as f:
        pkl.dump(train_model(model_and_hyperparameters, "/root/Project/jax_machinelearning/datasets/8mfsk/waveforms_CNO.pkl"), f)