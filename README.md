# TODO
## Better spectrum kernel
- center kernel
- do not normalize kernel? at least do it in a meaningful way
- standard spectrum kernel: no notion of mutations, O(l)
- substring kernel: all sub-sequences (not continuous) but decays exponentially with number of gaps, O(kl²)
- mismatch kernel: like spectrum but with m mismatches
- levenshtein: on whole sequence, O(l²)
- local alignment kernel: seems very similar to levenshtein, pd when taking exponential and summing over paths 
(in practice take the log because the values vary too wildly, but not pd anymore), O(l²) as well
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
## Edit distance
- reimplement in Cython
- read paper: seems to be a kernel
## Other
- optimize K to make points with similar labels close
### Brand new kernel (Probably will never be)
- dependent on position
- emphasis on center?
- a priori not symmetric left/right
- not symmetric A/T/C/G
- robust to addition/deletion
- proba kernel with HMM modeling
