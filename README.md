# TODO
## Better spectrum kernel
- center kernel
- do not normalize kernel? at least do it in a meaningful way
- be robust to addition/deletion: mismatch kernel
## SVMs
- Squared hinge vs hinge
- Crossval C: renormalize by 1/n
## Gradients
- weighted combination of the k-grams
- more complex weights to the grams, (something like $w_s = w_{\mathrm{len}(s)} \sum_{c \in s} w_c$)
- do not count the number of k-grams, but add weights based on the position of the k-grams
- levenshtein/mismatch kernel is a dot product (type of op, weight)
- gradients for SVM or ridge regression if unstable
- drop folds if too costly
- if still too costly: explicit freeing
## Other
- optimize K to make points with similar labels close
### Brand new kernel (Probably will never be)
- dependent on position
- emphasis on center?
- a priori not symmetric left/right
- not symmetric A/T/C/G
- robust to addition/deletion
- proba kernel with HMM modeling
