In root folder,

To setup the required packages (`numpy`, `scipy`, `cython` and `tqdm`)  run:
`pip install -r requirements.txt` then `conda install cvxopt`.

To compile the Cython code run:
`python setup.py build_ext --inplace`.

Then, to generate the solution, run:
`python start.py`.