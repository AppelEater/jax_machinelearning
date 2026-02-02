from LRUandJaxLIB import *
from alive_progress import alive_bar
from datetime import datetime
import optax

#
# Input parameters
#

# Dataset file path
dataset_file_path = "./datasets/8mfsk/absolute_doppler_waveforms_CNO_[14.2],[16.67]_and90_samprate_2000_1736935556.6539564.pkl"

# Test Ratio
test_ratio = 0.8

# Output folder
output_folder_path ="./results_new/grid_search34"

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
def train_model(config, model_parameters, training_data, key):
    """
    Train the model with the given hyperparameters and data

    model_and_hyper_parameters : dictionary
    data_file_path : string

    return : None
    """

    epoch_model_parameters = []
    epoch_model_parameters.append(model_parameters)  # initial snapshot
    
    train_sequences, train_labels, test_sequences, test_labels = training_data

    # Setup the optimizer
    if config["optimizer"] == "AdamW":
        if config["learning_rate"]["type"] == "Constant":
            lr = config["learning_rate"]["value"]
        elif config["learning_rate"]["type"] == "Cosine":
            lr = optax.cosine_decay_schedule(
                init_value = config["learning_rate"]["value"],
                decay_steps=config["epochs"] * len(train_sequences),
            )
        else:
             raise ValueError("Wrong learning rate decay type")
        optimizer = optax.adamw(learning_rate=lr, weight_decay=0.05)
    elif config["optimizer"] == "Adam":
        optimizer = optax.adam(learning_rate=config["learning_rate"]["value"])

    opt_state = optimizer.init(model_parameters)

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    prob = config["dropout"]

    # Train the model
    for epoch in range(config["epochs"]):
        
        with alive_bar(len(train_sequences)) as bar:
            for i in range(len(train_sequences)):
                key, subkey = jax.random.split(key)
                grads = model_grad(train_sequences[i], train_labels[i], model_parameters, prob, subkey)
                updates, opt_state = optimizer.update(grads, opt_state, model_parameters)
                model_parameters = optax.apply_updates(model_parameters, updates)
                key, _ = jax.random.split(key)
                bar()

        # Calculate the loss and accuracy
        train_acc.append(np.mean([accuracy(jnp.array(x), jnp.array(y), model_parameters, 0, key) for x, y in zip(train_sequences, train_labels)]))
        test_acc.append(np.mean([accuracy(jnp.array(x), jnp.array(y), model_parameters, 0, key) for x, y in zip(test_sequences, test_labels)]))
        train_loss.append(np.mean([loss_fn(jnp.array(x), jnp.array(y), model_parameters,0, key) for x, y in zip(train_sequences, train_labels)]))
        test_loss.append(np.mean([loss_fn(jnp.array(x), jnp.array(y), model_parameters, 0, key) for x, y in zip(test_sequences, test_labels)]))
        print("Test acc.:", ", ".join(f"{x:.3f}" for x in test_acc))
        epoch_model_parameters.append(model_parameters)

    # Save the model
    results = {
        "accuracy": {
            "Training accuracy" : train_acc,
            "Testing accuracy" : test_acc,
        },
        "loss": {
            "Training loss" : train_loss,
            "Testing loss" : test_loss,
        },
    }

    return epoch_model_parameters, results

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
        LRU_sub_1 = init_lru_parameters(mem_size, 257, r_min =0.9, r_max=0.999)
        LRU_Mixer_1 = init_mlp_parameters([257,257,257])
        LRU_sub_2 = init_lru_parameters(mem_size, 257, r_min =0.9, r_max=0.999)
        LRU_Mixer_2 = init_mlp_parameters([257,257])
        Decoding_layer = init_mlp_parameters([257,100,9])

        for idx, learning_rate in enumerate(learning_rates):
            for drop_out in dropout_list:

                config = {
                    "dropout": drop_out,
                    "batch_size": 100,
                    "epochs": 20,
                    "optimizer": "AdamW",
                    "encoding": "STFT",
                    "encoding_options": {
                        "Window Filter" : 'hann',
                        "Length" : 512,
                        "Hop" : 256,
                    },
                    "loss_function": "CrossEntropy",
                    "metric": "Accuracy",
                    "learning_rate": {
                        "type":  "Constant",
                        "value" : learning_rate,
                    },
                }

                # Load the data
                training_data, metadata = load_data(dataset_file_path, config["batch_size"],
                                                    test_ratio=test_ratio,
                                                    encoding=config["encoding"],
                                                    encoding_options=config.get("encoding_options", None))

                config["dataset"] = {
                    "file_path": dataset_file_path,
                    "sampling_rate": metadata["sampling_rate"],
                    "tone_duration": metadata["tone_duration"],
                    "tones": metadata["tones"],
                    "noise": metadata["noise"],
                    "num_waveforms_per_class": metadata["num_waveforms_per_class"],
                    "CNO_list": metadata["CNO_list"],
                    "doppler_uncertainty_list": metadata["doppler_uncertainty_list"],
                    "doppler_rate_uncertainty": metadata["doppler_rate_uncertainty"],
                    "test_ratio": test_ratio,
                }

                init_parameters = (Encoding_layer, [LRU_sub_1, LRU_sub_2] , [LRU_Mixer_1, LRU_Mixer_2], Decoding_layer)

                parameters, results = train_model(config, init_parameters, training_data, key)

                output =  {
                    "schema_version": "1.0",
                    "config": config,
                    "parameters": parameters,
                    "results": results,
                }

                with open(f"{output_folder_path}/results{idx} time {datetime.now():%Y-%m-%d %H-%M-%S.%f}.pkl", "wb") as f:                
                    pkl.dump(output, f)
