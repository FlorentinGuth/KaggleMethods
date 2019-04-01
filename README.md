In root folder,

To setup the required packages (`numpy`, `scipy`, `cvxopt` and `tqdm`)  run:
`pip install -r requirements.txt`.

To compile the Cython code run:
`python src/setup.py build_ext --inplace`.

Then, to generate the solution, run:
`python start.py`.