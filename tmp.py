import pickle
import numpy as np
import matplotlib.pyplot as plt

pckl_path = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/cacheio_v11/ss_lstm_format_argo_forecasting_v11_train_0_9999.pickle"

pickle_in = open(pckl_path, "rb")
[scene_input_pckl, social_input_pckl, person_input_pckl, expected_output_pckl] = pickle.load(pickle_in)

# %%
scene_input_pckl.shape
img_path = scene_input_pckl[0][0]
img_path
plt.imshow(np.load(img_path))
plt.show()
