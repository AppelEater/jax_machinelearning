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
    print("Before load")

    # Load the data
    train_sequences, train_labels, test_sequences, test_labels = load_data(data_file_path, model_and_hyper_parameters_for_function["batch_size "], 9)

    print(test_labels.shape)

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

# Dataset file path
dataset_file_path = "/root/Project/jax_machinelearning/datasets/8mfsk/accu_test_waveforms_CNO_[14.2],[16.666666666666668]_and[0.05]_samprate_1600.pkl"




# Batch size
batch_sizes = [15]

# Learning rate
boundaries = [7200, 9600, 12000]  # Steps where LR changes
values = [0.0002, 0.00015, 0.0001, 0.00005]  # LR for each interval
learning_rates = [0.0002, 0.00025, 0.00015] 
# optax.piecewise_constant_schedule(
#     init_value=0.0002,
#     boundaries_and_scales=dict(zip(boundaries, values[1:])),
# )

mem_size_set = [256]

# FROM GRID SEARCH SEVEN THE DICTIONARY CHANGES, LEARNING RATE CAN NO LONGER BE SAVED BY IT SELF AS IT CAN BE A SCHEDULE


# Define the hyperparameters and model
for i in range(3): 
    for idx, learning_rate in enumerate(learning_rates):
        for mem_size in mem_size_set:
            # Define the model
            Encoding_layer = init_mlp_parameters([1,3,5,10])
            LRU_sub_1 = init_lru_parameters(mem_size, 10, r_min =0.9, r_max=0.999)
            LRU_nonlinear_part = init_mlp_parameters([10,10,10])
            Decoding_layer = init_mlp_parameters([10,10,9])
            
            model_and_hyperparameters = {"Model Parameters" : (Encoding_layer, LRU_sub_1, LRU_nonlinear_part, Decoding_layer),
                                        "Learning Rate" : learning_rate,
                                        "batch_size " : 15,
                                        "epochs" : 20,
                                        "optimizer" : "Adam",
                                        "loss_function" : "CrossEntropy",
                                        "metric" : "Accuracy",
                                        "training dataset circumstance" : "MFSK signal, with 100 Hz spacing and 1.6kHz sampling rate with different learning rates and init phases pi/10",
                                        "File Path" : dataset_file_path,
                                        "Learning Schedule": { 
                                                    "Schedule Type": "Constant",
                                                    "Learning_rate": learning_rate
                                        } }

            with open(f"grid_search14/results{idx} time {datetime.now()}.pkl", "wb") as f:
                pkl.dump({k:v for k,v in train_model(model_and_hyperparameters, dataset_file_path).items() if k != "Learning Rate"}, f)