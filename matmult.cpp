// vim:set ft=cpp sw=4 ts=4 sts=4 noet:
#include <iostream>
#include <vector>
#include <cstdlib>
#include <type_traits>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sstream>

void assert_fail(const char * str) {
	std::cerr << str << std::endl;
	std::exit(1);
}

#define assert(cond) do { \
	if (!(cond)) { \
		assert_fail("Assertion failed: " #cond); \
	} \
} while (false)

template <typename T>
class matrix {
public:
	typedef T value_type;

	matrix(size_t rows=0, size_t columns=0) {
		resize(rows, columns);
	}

	size_t rows() const { return m_rows; }
	size_t columns() const { return m_columns; }
	size_t data_width() const { return m_columns; }
	void resize(size_t n, size_t m) {
		m_rows = n;
		m_columns = m;
		std::vector<T> data(n*m);
		std::swap(data, m_data);
	}

	T & at(size_t i, size_t j) {
		return m_data[i * m_columns + j];
	}

	const T & at(size_t i, size_t j) const {
		return m_data[i * m_columns + j];
	}

	T * get() {
		return &m_data[0];
	}

private:
	std::vector<T> m_data;
	size_t m_rows;
	size_t m_columns;
};

template <typename T, bool Mut>
class matrix_view {
public:
	typedef T value_type;

private:
	typedef typename std::conditional<Mut, T *, const T *>::type data_type;
	typedef typename std::conditional<Mut, T &, const T &>::type reference_type;

public:
	matrix_view(data_type data, size_t rows, size_t columns, size_t dataWidth)
		: m_data(data)
		, m_rows(rows)
		, m_columns(columns)
		, m_dataWidth(dataWidth)
	{
	}

	size_t rows() const { return m_rows; }
	size_t columns() const { return m_columns; }
	size_t data_width() const { return m_dataWidth; }
	void resize(size_t, size_t) {
		assert_fail("Attempt to resize a matrix_view");
	}

	reference_type at(size_t i, size_t j) {
		return m_data[i * m_dataWidth + j];
	}

	const T & at(size_t i, size_t j) const {
		return m_data[i * m_dataWidth + j];
	}

private:
	data_type m_data;
	size_t m_rows;
	size_t m_columns;
	size_t m_dataWidth;
};

template <typename T, typename child, bool mut>
class matrix_file {
private:
	child & self() { return *static_cast<child *>(this); }

public:
	matrix_file(const char * fileName, size_t n, size_t m)
		: m_rows(n)
		, m_columns(m)
	{
		m_fd = open(fileName, self().open_flags(), 0644);
		self().post_open();
		m_data = (T *) mmap(NULL, m_rows*m_columns*sizeof(T), self().prot_flags(), MAP_SHARED, m_fd, 0);
	}

	~matrix_file() {
		munmap(m_data, m_rows * m_columns * sizeof(T));
		close(m_fd);
	}

	matrix_view<T, mut> view() {
		return matrix_view<T, mut>(m_data, m_rows, m_columns, m_columns);
	}

protected:
	int m_fd;
	T * m_data;
	size_t m_rows;
	size_t m_columns;
};

template <typename T>
class matrix_file_input : public matrix_file<T, matrix_file_input<T>, false> {
private:
	friend class matrix_file<T, matrix_file_input<T>, false>;
	int open_flags() { return O_RDONLY; }
	int prot_flags() { return PROT_READ; }
	void post_open() {}

public:
	using matrix_file<T, matrix_file_input<T>, false>::matrix_file;
};

template <typename T>
class matrix_file_output : public matrix_file<T, matrix_file_output<T>, true> {
private:
	friend class matrix_file<T, matrix_file_output<T>, true>;
	int open_flags() { return O_RDWR | O_CREAT | O_TRUNC; }
	int prot_flags() { return PROT_READ | PROT_WRITE; }
	void post_open() {
		ftruncate(this->m_fd, sizeof(T) * this->m_rows * this->m_columns);
	}

public:
	using matrix_file<T, matrix_file_output<T>, true>::matrix_file;
};

template <typename M, typename T=typename M::value_type>
matrix_view<T, true>
slice(M & a, size_t i1, size_t i2, size_t j1, size_t j2) {
	return matrix_view<T, true>(&a.at(i1, j1), i2-i1, j2-j1, a.data_width());
}

template <typename M, typename T=typename M::value_type>
matrix_view<T, false>
slice(const M & a, size_t i1, size_t i2, size_t j1, size_t j2) {
	return matrix_view<T, false>(&a.at(i1, j1), i2-i1, j2-j1, a.data_width());
}

struct multiply_naive {
	template <typename A, typename B, typename AB>
	void operator()(const A & a, const B & b, AB && ab) {
		const size_t n = a.rows();
		const size_t m = a.columns();
		assert(m == b.rows());
		const size_t p = b.columns();
		// a is n*m, b is m*p, ab is n*p
		// ab[i, j] = sum of a[i, k] * b[k, j]
		for (size_t i = 0; i < n; ++i) {
			for (size_t j = 0; j < p; ++j) {
				for (size_t k = 0; k < m; ++k) {
					ab.at(i, j) += a.at(i, k) * b.at(k, j);
				}
			}
		}
	}
};

struct multiply_recursive {
	template <typename A, typename B, typename AB>
	void operator()(const A & a, const B & b, AB && ab) {
		const size_t n = a.rows();
		assert(n == ab.rows());
		const size_t m = a.columns();
		assert(m == b.rows());
		const size_t p = b.columns();
		assert(p == ab.columns());
		if (n <= 1 || m <= 1 || p <= 1 || n*m*p <= 1000) {
			multiply_naive()(a, b, ab);
			return;
		}
		const size_t max = std::max(std::max(n, m), p);
		if (m == max) {
			size_t h = m / 2;
			(*this)(slice(a, 0, n, 0, h), slice(b, 0, h, 0, p), ab);
			(*this)(slice(a, 0, n, h, m), slice(b, h, m, 0, p), ab);
		} else if (n == max) {
			size_t h = n / 2;
			(*this)(slice(a, 0, h, 0, m), b, slice(ab, 0, h, 0, p));
			(*this)(slice(a, h, n, 0, m), b, slice(ab, h, n, 0, p));
		} else if (p == max) {
			size_t h = p / 2;
			(*this)(a, slice(b, 0, m, 0, h), slice(ab, 0, n, 0, h));
			(*this)(a, slice(b, 0, m, h, p), slice(ab, 0, n, h, p));
		}
	}
};

template <typename A, typename B>
void matrix_copy(A && a, B & b) {
	const size_t n = a.rows();
	const size_t m = a.columns();
	for (size_t i = 0; i < n; ++i) {
		std::copy(&a.at(i, 0), &a.at(i, 0) + m, &b.at(i, 0));
	}
}

struct op_add {
	template <typename T>
	void operator()(const T & x, T & y) const {
		y += x;
	}
};

struct op_sub {
	template <typename T>
	void operator()(const T & x, T & y) const {
		y -= x;
	}
};

template <typename A, typename B, typename Op>
void matrix_op(A && a, B & b, Op op) {
	const size_t n = a.rows();
	const size_t m = a.columns();
	for (size_t i = 0; i < n; ++i) {
		auto x = &a.at(i, 0);
		auto y = &b.at(i, 0);
		for (size_t j = 0; j < m; ++j) op(*x++, *y++);
	}
}

template <typename A, typename B>
void matrix_add(A && a, B & b) {
	matrix_op(a, b, op_add());
}

template <typename A, typename B>
void matrix_sub(A && a, B & b) {
	matrix_op(a, b, op_sub());
}

struct multiply_strassen_impl {
	template <typename A, typename B, typename AB>
	void operator()(const A & a, const B & b, AB && ab) {
		typedef typename A::value_type T;

		const size_t n = a.rows();
		if (n <= 512) {
			multiply_recursive()(a, b, ab);
			return;
		}
		std::cout << "strassen(" << n << ")" << std::endl;
		const size_t h = n / 2;

		auto a11 = slice(a, 0, h, 0, h);
		auto a12 = slice(a, 0, h, h, n);
		auto a21 = slice(a, h, n, 0, h);
		auto a22 = slice(a, h, n, h, n);
		auto b11 = slice(b, 0, h, 0, h);
		auto b12 = slice(b, 0, h, h, n);
		auto b21 = slice(b, h, n, 0, h);
		auto b22 = slice(b, h, n, h, n);
		auto c11 = slice(ab, 0, h, 0, h);
		auto c12 = slice(ab, 0, h, h, n);
		auto c21 = slice(ab, h, n, 0, h);
		auto c22 = slice(ab, h, n, h, n);

		matrix<T> m1(h, h);
		matrix<T> m2(h, h);
		matrix<T> m3(h, h);
		matrix<T> m4(h, h);
		matrix<T> m5(h, h);
		matrix<T> m6(h, h);
		matrix<T> m7(h, h);

		matrix<T> t1(h, h);
		matrix<T> t2(h, h);

		std::cout << "compute m1" << std::endl;
		matrix_copy(a11, t1);
		matrix_add(a22, t1);
		matrix_copy(b11, t2);
		matrix_add(b22, t2);
		multiply_strassen_impl()(t1, t2, m1);

		std::cout << "compute m2" << std::endl;
		matrix_copy(a21, t1);
		matrix_add(a22, t1);
		multiply_strassen_impl()(t1, b11, m2);

		std::cout << "compute m3" << std::endl;
		matrix_copy(b12, t2);
		matrix_sub(b22, t2);
		multiply_strassen_impl()(a11, t2, m3);

		std::cout << "compute m4" << std::endl;
		matrix_copy(b21, t2);
		matrix_sub(b11, t2);
		multiply_strassen_impl()(a22, t2, m4);

		std::cout << "compute m5" << std::endl;
		matrix_copy(a11, t1);
		matrix_add(a12, t1);
		multiply_strassen_impl()(t1, b22, m5);

		std::cout << "compute m6" << std::endl;
		matrix_copy(a21, t1);
		matrix_sub(a11, t1);
		matrix_copy(b11, t2);
		matrix_add(b12, t2);
		multiply_strassen_impl()(t1, t2, m6);

		std::cout << "compute m7" << std::endl;
		matrix_copy(a12, t1);
		matrix_sub(a22, t1);
		matrix_copy(b21, t2);
		matrix_add(b22, t2);
		multiply_strassen_impl()(t1, t2, m7);

		std::cout << "compute c" << std::endl;
		matrix_copy(m1, c11);
		matrix_add(m4, c11);
		matrix_sub(m5, c11);
		matrix_add(m7, c11);
		matrix_copy(m3, c12);
		matrix_add(m5, c12);
		matrix_copy(m2, c21);
		matrix_add(m4, c21);
		matrix_copy(m1, c22);
		matrix_sub(m2, c22);
		matrix_add(m3, c22);
		matrix_add(m6, c22);
	}
};

struct multiply_strassen {
	template <typename A, typename B, typename AB>
	void operator()(const A & a, const B & b, AB && ab) {
		typedef typename A::value_type T;

		const size_t n = a.rows();
		const size_t m = a.columns();
		assert(m == b.rows());
		const size_t p = b.columns();
		const size_t max = std::max(std::max(n, m), p);
		size_t x = 1;
		while (x < max) x += x;
		std::cout << "strassen(" << n << ", " << m << ", " << p << ") has x == " << x << std::endl;
		if (n == x && m == x && p == x) {
			multiply_strassen_impl()(a, b, ab);
			return;
		}
		matrix<T> aa;
		matrix<T> bb;
		matrix<T> res;
		aa.resize(x, x);
		bb.resize(x, x);
		res.resize(x, x);
		matrix_copy(a, aa);
		matrix_copy(b, bb);
		multiply_strassen_impl()(aa, bb, res);
		matrix_copy(slice(res, 0, n, 0, p), ab);
	}
};

template <typename A, typename B, typename AB>
void multiply_mode(const std::string & mode, const A & a, const B & b, AB && ab) {
	if (mode == "recursive") {
		multiply_recursive()(a, b, ab);
	} else if (mode == "naive") {
		multiply_naive()(a, b, ab);
	} else if (mode == "strassen") {
		multiply_strassen()(a, b, ab);
	} else {
		assert_fail("What mode do you want?");
	}
}

int main(int argc, char ** argv) {
	if (argc < 5)
		assert_fail("Not enough arguments! Need: mode n m p");

	std::string mode;
	size_t n, m, p;

	char ** arg = argv + 1;

	mode = *arg++;
	std::stringstream(*arg++) >> n;
	std::stringstream(*arg++) >> m;
	std::stringstream(*arg++) >> p;

	multiply_mode(mode,
		matrix_file_input<double>("a", n, m).view(),
		matrix_file_input<double>("b", m, p).view(),
		matrix_file_output<double>("ab", n, p).view());
	return 0;
}
