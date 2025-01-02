import glob
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

gloober = glob.glob("grid_search10/*")


##### Figures to be plotted #####
#    - Best accuracy plot, legend should include batch size and learning rate
#      secondary axis is validation loss

fig, ax = plt.subplots()

ax2 = ax.twinx()

best_yet_acc = [0, 0]

list_of_best_acc = []

print(len(gloober))
print(gloober)
# Plot accuracy and 
for idx, glib in enumerate(gloober):
    print(idx)
    try:
        with open(glib, "rb") as f:
            dictionary = pkl.load(f)
            if np.max(dictionary["Accuracy Measurements"]["Testing accuracy"]) > best_yet_acc[0]:
                best_yet_acc[0] = np.max(dictionary["Accuracy Measurements"]["Testing accuracy"])
                best_yet_acc[1] = idx
            list_of_best_acc.append(np.max(dictionary["Accuracy Measurements"]["Testing accuracy"]))

    except EOFError:
        print("File still open")
print(best_yet_acc)


dict = {}
with open(gloober[best_yet_acc[1]], "rb") as f:
    dict = pkl.load(f)

print(dict.keys())

print(dict["Dropout"])

ax.plot(dict["Accuracy Measurements"]["Testing accuracy"], label = f'Learning rate] & Batch size {dict["batch_size "] }')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy %")

fig.legend()
ax2.plot(dict["Loss Measurements"]["Testing loss"], color="r")
ax2.set_ylabel("Testing Loss", color = "r")

plt.savefig("Second grid search.png")

# Heat map of max test accuracy, with the axes learning rate and batch size
learning_rates = jnp.logspace(-4,-7,5)
batch_sizes = np.arange(5,25,5)



heat_map_array = np.array(list_of_best_acc).reshape(len(list_of_best_acc),1)

fig, ax = plt.subplots()

im = ax.imshow(heat_map_array)

fig.colorbar(im, ax=ax, label= "Colorbar")
plt.savefig(f"Second Heat map grid search")