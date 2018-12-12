import matplotlib.pyplot as plt
import numpy as np

# def read_log(filename):
#     bleu, accy = []
#     with open(filename, 'r') as f:
#         for line in f:
#             if line.startswith():


# fig = plt.figure()
# ax = plt.axes()

# x = [i for i in range(1, 16)]
# mrt_bleu_bleu = [34.2188, 34.1289, 34.0965, 33.9093, 34.1850, 34.3518, 34.4065, 34.4368, 34.4438, 34.4000, 34.4420, 34.4656, 34.4219, 34.4589, 34.4442]
# mrt_nli_bleu = [34.1224, 33.9067, 32.4719, 30.5378, 30.8832, 29.5363, 30.1755, 29.2405, 29.9605, 28.5234, 28.0710, 28.8168, 28.4575, 28.4361, 28.2187]
# mrt_combined_bleu = [33.9144, 34.1088, 34.1373, 34.0198, 33.9711, 33.9723, 33.8103, 33.5894, 33.3995, 33.4059, 33.3904, 33.1942, 33.2272, 33.0554, 33.0312]
# ax.plot(x, mrt_bleu_bleu, label='MRT-BLEU')
# ax.plot(x, mrt_nli_bleu, label='MRT-NLI')
# ax.plot(x, mrt_combined_bleu, label='MRT-COMBINED')
# ax.set_facecolor((232/255, 235/255, 241/255))
# ax.legend()

# plt.plot()
# plt.title('epoch number vs BLEU score')
# plt.xlabel('epoch number')
# plt.ylabel('BLEU score')
# fig.savefig('test.png', bbox_inches='tight', pad_inches=0)
# # plt.show()

# fig = plt.figure()
# ax = plt.axes()
# mrt_bleu_accy = [0.5625, 0.5625, 0.5615, 0.5654, 0.5596, 0.5615, 0.5449, 0.5449, 0.5449, 0.5469, 0.5410, 0.5439, 0.5469, 0.5459, 0.5537]
# mrt_nli_accy = [0.5661, 0.5700, 0.5407, 0.5573, 0.5524, 0.5426, 0.5690, 0.5690, 0.5485, 0.5671, 0.5612, 0.5690, 0.5651, 0.5846, 0.5602]
# mrt_combined_accy = [0.5485, 0.5514, 0.5397, 0.5544, 0.5446, 0.5465, 0.5632, 0.5573, 0.5505, 0.5426, 0.5456, 0.5495, 0.5514, 0.5553, 0.5563]
# ax.plot(x, mrt_bleu_accy, label='MRT-BLEU')
# ax.plot(x, mrt_nli_accy, label='MRT-NLI')
# ax.plot(x, mrt_combined_accy, label='MRT-COMBINED')
# ax.set_facecolor((232/255, 235/255, 241/255))
# ax.legend()

# plt.plot()
# plt.title('epoch number vs XNLI accuracy')
# plt.xlabel('epoch number')
# plt.ylabel('XNLI accuracy')
# plt.show()

# n_groups = 3

# accy_base = [0.55828221, 0.80827887, 0.3106383]
# accy_bleu = [0.52147239, 0.82352941, 0.34255319]
# accy_nli = [0.599182, 0.61437908, 0.59574468]
# accy_combined = [0.59100204, 0.70588235, 0.45744681]

# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.15

# opacity = 0.4
# error_config = {'ecolor': '0.3'}

# rects1 = ax.bar(index, accy_base, bar_width,
#                 alpha=opacity, color='b', label='Base')
# rects1 = ax.bar(index + bar_width * 1, accy_bleu, bar_width,
#                 alpha=opacity, color='g', label='MRT-BLEU')
# rects1 = ax.bar(index + bar_width * 2, accy_nli, bar_width,
#                 alpha=opacity, color='y', label='MRT-NLI')
# rects1 = ax.bar(index + bar_width * 3, accy_combined, bar_width,
#                 alpha=opacity, color='r', label='MRT-COMBINED')

# ax.set_xlabel('Label Category')
# ax.set_ylabel('Accuracy')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('Contradiction', 'Neutral', 'Entailment'))
# ax.legend()

# plt.ylim(0.25, 0.85)
# fig.tight_layout()
# plt.show()

n_groups = 3

accy_base = [0.51677852, 0.50359712, 0.58474576]
accy_nli = [0.59731544, 0.57553957, 0.66949153]
accy_combined = [0.59060403, 0.53956835, 0.65254237]
accy_bleu = [0.54362416, 0.49640288, 0.55932203]

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.15

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, accy_base, bar_width,
                alpha=opacity, color='b', label='Base')
rects1 = ax.bar(index + bar_width * 1, accy_bleu, bar_width,
                alpha=opacity, color='g', label='MRT-BLEU')
rects1 = ax.bar(index + bar_width * 2, accy_nli, bar_width,
                alpha=opacity, color='y', label='MRT-NLI')
rects1 = ax.bar(index + bar_width * 3, accy_combined, bar_width,
                alpha=opacity, color='r', label='MRT-COMBINED')

ax.set_xlabel('Genre')
ax.set_ylabel('Accuracy')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Government', 'Oup', 'Telephone'))
ax.legend()

plt.ylim(0.4, 0.75)
fig.tight_layout()
plt.show()