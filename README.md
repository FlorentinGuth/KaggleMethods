# TODO
## Better spectrum kernel
- center kernel
- do not normalize kernel?
- weighted combination of the k-grams (to learn)
- more complex weights to the grams, (something like $w_s = w_{\mathrm{len}(s)} \sum_{c \in s} w_c$)
- do not count the number of k-grams, but add weights based on the position of the k-grams
- be robust to addition/deletion
## Brand new kernel
- dependent on position
- emphasis on center?
- a priori not symmetric left/right
- not symmetric A/T/C/G
- robust to addition/deletion
- proba kernel with HMM modeling
## Misc
- investigate the common subsequences stuff
- t-SNE / Kernel PCA (doesn't work?)
