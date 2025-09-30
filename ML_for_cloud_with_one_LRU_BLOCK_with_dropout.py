from LRUandJaxLIB import *
from alive_progress import alive_bar
from datetime import datetime
import optax

#
# Input parameters
#

# Dataset file path
dataset_file_path = "./datasets/8mfsk/absolute_doppler_waveforms_CNO_[14.2],[16.67]_and90_samprate_2000_1736935556.6539564.pkl"

# Output folder
output_folder_path ="./results/grid_search34"

# Batch size
batch_sizes = [100]

# Learning rate
learning_rates = [0.0001, 0.00009] 
# boundaries = [7200, 9600, 12000]  # Steps where LR changes
# values = [0.0002, 0.00015, 0.0001, 0.00005]  # LR for each interval

# Memory size
LRU_memory_list = [526]

# Dropout probability
dropout_list = [0.025, 0.05]

# -----------------------------------------------------------------------------

# Make a train model function
#
def train_model(model_and_hyper_parameters_for_function, data_file_path, key):
    """
    Train the model with the given hyperparameters and data

    model_and_hyper_parameters : dictionary
    data_file_path : string

    return : None
    """

    # Load the data
    train_sequences, train_labels, test_sequences, test_labels = load_data(data_file_path, model_and_hyper_parameters_for_function["batch_size "], 9, 0.8,
                                                                            model_and_hyper_parameters_for_function["Encoding"],
                                                                            model_and_hyper_parameters_for_function.get("Encoding Options", None))

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

    prob = model_and_hyper_parameters_for_function["Dropout"]

    # Train the model
    for epoch in range(model_and_hyper_parameters_for_function["epochs"]):
        
        with alive_bar(len(train_sequences)) as bar:
            for i in range(len(train_sequences)):
                grads = model_grad2(train_sequences[i], train_labels[i], model_parameters, prob, key, True)
                updates, opt_state = optimizer.update(grads, opt_state, model_parameters)
                model_parameters = optax.apply_updates(model_parameters, updates)
                key, _ = jax.random.split(key)
                bar()

        # Calculate the loss and accuracy
        train_acc.append(np.mean([accuracy2(jnp.array(x), jnp.array(y), model_parameters, 0,key,False) for x, y in zip(train_sequences, train_labels)]))
        test_acc.append(np.mean([accuracy2(jnp.array(x), jnp.array(y), model_parameters, 0,key,False) for x, y in zip(test_sequences, test_labels)]))
        train_loss.append(np.mean([loss_fn2(jnp.array(x), jnp.array(y), model_parameters, 0,key,False) for x, y in zip(train_sequences, train_labels)]))
        test_loss.append(np.mean([loss_fn2(jnp.array(x), jnp.array(y), model_parameters, 0,key, False) for x, y in zip(test_sequences, test_labels)]))
        print(f"Test acc {test_acc}")
        epoch_model_parameters.append(model_parameters)

    # Save the model
    model_and_hyper_parameters_for_function["Accuracy Measurements"] = {"Training accuracy" : train_acc,
                                                                        "Testing accuracy" : test_acc}
    model_and_hyper_parameters_for_function["Loss Measurements"] = {"Training loss" : train_loss,
                                                                    "Testing loss" : test_loss}

    model_and_hyper_parameters_for_function["Model Parameters"] = epoch_model_parameters

    return model_and_hyper_parameters_for_function

# -----------------------------------------------------------------------------

# Key
key = jax.random.key(135)

# optax.piecewise_constant_schedule(
#     init_value=0.0002,
#     boundaries_and_scales=dict(zip(boundaries, values[1:])),
# )

# Define the hyperparameters and model

for mem_size in LRU_memory_list :
    for i in range(4):
        # Define the model
        Encoding_layer = init_mlp_parameters([257,257])
        LRU_sub = init_lru_parameters(mem_size, 257, r_min =0.9, r_max=0.999)
        LRU_Mixer = init_mlp_parameters([257,257,257])
        Decoding_layer = init_mlp_parameters([257,100,9])

        for idx, learning_rate in enumerate(learning_rates):
            for drop_out in dropout_list:
                model_and_hyperparameters = {"Model Parameters" : (Encoding_layer, LRU_sub, LRU_Mixer, Decoding_layer),
                                            "Learning Rate" : learning_rate,
                                            "batch_size " : 100,
                                            "epochs" : 20,
                                            "optimizer" : "AdamW",
                                            "Encoding" : "STFT",
                                            "Encoding Options" : {
                                                                    "Window Filter" : 'hann',
                                                                    "Length" : 512,
                                                                    "Hop" : 256
                                                                  },
                                            "Dropout" : drop_out,
                                            "loss_function" : "CrossEntropy",
                                            "metric" : "Accuracy",
                                            "training dataset circumstance" : "MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT",
                                            "File Path" : dataset_file_path,
                                            "Learning Schedule": {  
                                                        "Schedule Type": "Constant",
                                                        "Value" : learning_rate
                                            } }

                with open(f"{output_folder_path}/results{idx} time {datetime.now():%Y-%m-%d %H-%M-%S.%f}.pkl", "wb") as f:                
                    pkl.dump({k:v for k,v in train_model(model_and_hyperparameters, dataset_file_path, key).items() if k != "Learning Rate"}, f)
