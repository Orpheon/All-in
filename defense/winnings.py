import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('whitegrid')


plt.figure(figsize=(16, 10))
plt.title("Cash earned per game")

dist = np.random.beta(a=2, b=5, size=10000)
max = np.max(dist)
min = np.min(dist)
dist = 200 * (((dist - min) / (max - min)) * 6 - 1)
ax = sns.kdeplot(dist, shade=True)
plt.savefig('winnings_1.png', bbox_inches='tight')

fontsize = 48
spacing = 0.14
orig_y = 0.78
orig_x = 0.6

mean = np.mean(dist)
plt.axvline(mean, color='#d30000')
plt.text(orig_x, orig_y, "Mean", color='#d30000', fontsize=fontsize, transform=ax.transAxes)
plt.savefig('winnings_2.png', bbox_inches='tight')

orig_y -= spacing

median = np.median(dist)
plt.axvline(median, color='#53b128')
plt.text(orig_x, orig_y, "Median", color='#53b128', fontsize=fontsize, transform=ax.transAxes)
plt.savefig('winnings_3.png', bbox_inches='tight')

orig_y -= spacing

pctl = np.percentile(dist, 20)
plt.axvline(pctl, color='#ae24c8')
plt.text(orig_x, orig_y, "20-Percentile", color='#ae24c8', fontsize=fontsize, transform=ax.transAxes)
plt.savefig('winnings_4.png', bbox_inches='tight')
