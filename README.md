Efficient matrix multiplication.

The programs `matmult.cpp` and `matmult.py` each take four command line arguments:
mode, n, m, p,
and they each read the n-by-m matrix in the file `a`
and the m-by-p matrix in the file `b`
and write the multiplication to the file `ab`.

The format of the `a`, `b`, and `ab` files
is understood by the `matrix.py` helper program.

I believe that the `native` mode of `matmult.py`
and the `recursive` mode of `matmult.cpp`
are correct; the other modes are probably full of bugs.
