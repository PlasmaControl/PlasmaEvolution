import matplotlib.pyplot as plt
import pickle
import numpy as np

pkl_name = 'preprocessed_diiid_data_highip_val_losses.pkl'
with open(f'{pkl_name}', 'rb') as file:
    loss_dict = pickle.load(file)

titles = loss_dict.keys()
print(titles)
values = [loss_dict[key][0] for key in titles]
values[:] = np.mean(values[:], axis=2)


bplot = plt.boxplot(values, vert=True, patch_artist=True, widths=0.6)
for box in bplot['boxes']:
    box.set(facecolor='none', edgecolor='black')
plt.xticks(range(1, len(values) + 1), titles, rotation=20, fontsize=8)
plt.title('Losses for each model')
plt.ylabel('Loss per timestep')
plt.savefig('plots/box_whisker_test.svg')
