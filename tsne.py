import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

archetypes = torch.load('tensors/Z1.pt')
print(archetypes.shape)
archetypes = np.array(archetypes.reshape(32, 512))
print(archetypes.dtype)
archetypes_embedded = TSNE(n_components=2).fit_transform(archetypes)
print(archetypes_embedded.shape)
# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure()
plt.scatter(archetypes_embedded[:,0], archetypes_embedded[:,1])
# plt.show()
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=archetypes_embedded,
#     legend="full",
#     alpha=0.3
# )
plt.savefig("tsneZ1.png")