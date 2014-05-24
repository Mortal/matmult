import sys
import numpy as np
import struct

def read_matrix(file_name, n, m):
    fmt = struct.Struct('@' + 'd' * (n*m))
    with open(file_name, "rb") as fp:
        return np.asarray(fmt.unpack(fp.read(fmt.size))).reshape((n, m))

def write_matrix(file_name, res):
    n, m = res.shape
    fmt = struct.Struct('@' + 'd' * m)
    with open(file_name, "wb") as fp:
        for row in res:
            fp.write(fmt.pack(*row))

def multiply_naive(a, b):
    n, m = a.shape
    m, p = b.shape

    res = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            res[i, j] = np.dot(a[i, :], b[:, j])
    return res

def multiply_recursive(a, b):
    n, m = a.shape
    m, p = b.shape
    print("recurse(%s, %s, %s)" % (n, m, p))
    smallest = min(n, m, p)
    largest = max(n, m, p)
    if smallest <= 1 or n*m*p <= 512*512*512:
        return np.dot(a, b)
    res = np.zeros((n, p))
    if m == largest:
        h = m // 2
        res += multiply_recursive(a[:, 0:h], b[0:h, :])
        res += multiply_recursive(a[:, h:m], b[h:m, :])
    elif n == largest:
        h = n // 2
        res[0:h, :] += multiply_recursive(a[0:h, :], b)
        res[h:n, :] += multiply_recursive(a[h:n, :], b)
    elif p == largest:
        h = p // 2
        res[:, 0:h] += multiply_recursive(a, b[:, 0:h])
        res[:, h:p] += multiply_recursive(a, b[:, h:p])
    return res

def multiply_strassen_impl(a, b):
    n = a.shape[0]
    if n <= 1024:
        return multiply_recursive(a, b)
    print("strassen(%s)" % n)
    h = n // 2
    a11 = a[0:h, 0:h]
    a12 = a[0:h, h:n]
    a21 = a[h:n, 0:h]
    a22 = a[h:n, h:n]
    b11 = b[0:h, 0:h]
    b12 = b[0:h, h:n]
    b21 = b[h:n, 0:h]
    b22 = b[h:n, h:n]
    res = np.zeros((n, n))
    c11 = res[0:h, 0:h]
    c12 = res[0:h, h:n]
    c21 = res[h:n, 0:h]
    c22 = res[h:n, h:n]

    t1 = np.zeros((h, h))
    t2 = np.zeros((h, h))

    def add1(p, q):
        t1[:, :] = p
        t1[:, :] += q
        return t1

    def add2(p, q):
        t2[:, :] = p
        t2[:, :] += q
        return t2

    def sub1(p, q):
        t1[:, :] = p
        t1[:, :] -= q
        return t1

    def sub2(p, q):
        t2[:, :] = p
        t2[:, :] -= q
        return t2

    print("Compute m1")
    m1 = multiply_strassen_impl(add1(a11, a22), add2(b11, b22))
    print("Compute m2")
    m2 = multiply_strassen_impl(add1(a21, a22), b11)
    print("Compute m3")
    m3 = multiply_strassen_impl(a11, sub2(b12, b22))
    print("Compute m4")
    m4 = multiply_strassen_impl(a22, sub2(b21, b11))
    print("Compute m5")
    m5 = multiply_strassen_impl(add1(a11, a12), b22)
    print("Compute m6")
    m6 = multiply_strassen_impl(sub1(a21, a11), add2(b11, b12))
    print("Compute m7")
    m7 = multiply_strassen_impl(sub1(a12, a22), add2(b21, b22))

    print("Compute c")
    c11[:,:] = m1
    c11[:,:] += m4
    c11[:,:] -= m5
    c11[:,:] += m7
    c12[:,:] = m3
    c12[:,:] += m5
    c21[:,:] = m2
    c21[:,:] += m4
    c22[:,:] = m1
    c22[:,:] -= m2
    c22[:,:] += m3
    c22[:,:] += m6

    return res

def multiply_strassen(a, b):
    n, m = a.shape
    m, p = b.shape
    smallest = min(n, m, p)
    largest = max(n, m, p)
    x = 1
    while x < largest:
        x += x
    if smallest == x:
        return multiply_strassen_impl(a, b)
    aa = np.zeros((x, x))
    bb = np.zeros((x, x))
    aa[0:n, 0:m] = a
    bb[0:m, 0:p] = b
    return np.array(multiply_strassen_impl(aa, bb)[0:n, 0:p])

def main(mode, n, m, p):
    n, m, p = int(n), int(m), int(p)
    a = read_matrix('a', n, m)
    b = read_matrix('b', m, p)
    print("Compute multiplication")
    if mode == 'naive':
        res = multiply_naive(a, b)
    elif mode == 'native':
        res = np.dot(a, b)
    elif mode == 'recursive':
        res = multiply_recursive(a, b)
    elif mode == 'strassen':
        res = multiply_strassen(a, b)
    else:
        raise SystemExit("What mode do you want?")
    print("Write result")
    write_matrix("ab", res)

if __name__ == '__main__':
    main(*sys.argv[1:])
