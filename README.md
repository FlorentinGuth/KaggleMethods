In root folder,

To setup the required packages (`numpy`, `scipy`, `cython` and `tqdm`)  run:
`pip install -r requirements.txt` then `conda install cvxopt`.

To compile the Cython code run:
`python setup.py build_ext --inplace`.

Then, to generate the solution, run:
`python start.py`.

This will perform the following steps:
 - Compute all the required kernels
 (all spectrum, mismatch, and edit kernels);
 - Compute the validation score of each of these kernels,
 and save the results to `separate_kernels`;
 - Optimize the parameter T defining the weighted average;
 - Evaluate the final kernel and generate the final predictions.
 
 By default, it will use the 39 first spectrum kernels,
 12 first mismatch kernels, and the edit kernel.
 This can be reduced by changing the values at the end of `start.py`:
 parameters `spectrum_k`, `mismatch_k`, and `use_edit_kernel`.
 
 By default, it **will not** compute the validation scores,
 but use those we precomputed in `separate_kernels`.
 If you want to recompute everything, delete this `separate_kernels` file.
 Though, this step **is computationally expensive**.
 
 By default, it **will not** optimize the values of T,
 but use the values we found to be the bests.
 This can be changed by setting parameter `compute_T` to `true` at the end
 of `start.py`.