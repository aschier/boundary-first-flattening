//extern Common common;

#include <cstring>
#include <vector>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> cholmod_sparse;
typedef Eigen::Triplet<double> T;
typedef std::vector<Eigen::Triplet<double>> cholmod_triplet;

using std::memcpy;

inline cholmod_sparse* cholmod_l_spzeros(size_t m, size_t n, size_t nnz)
{
	cholmod_sparse mat(m, n);
	mat.resizeNonZeros(nnz);
	return new cholmod_sparse(mat);
}

inline cholmod_sparse* cholmod_l_triplet_to_sparse(cholmod_triplet* triplet, size_t m, size_t n)
{
	cholmod_sparse mat(m, n);
	mat.setFromTriplets(triplet->begin(), triplet->end());
	return new cholmod_sparse(mat);
}

inline cholmod_sparse* cholmod_l_speye(size_t m, size_t n)
{
	cholmod_sparse mat(m, n);
	auto min = std::min(m, n);
	mat.resizeNonZeros(min);

	cholmod_triplet triplets;
	for (int i = 0; i < min; ++i)
		triplets.emplace_back(T(i, i, 1.0));

	return new cholmod_sparse(mat);
}

inline cholmod_sparse* cholmod_l_transpose(cholmod_sparse* input)
{
	return new cholmod_sparse(input->transpose());
}

inline cholmod_dense* cholmod_l_sparse_to_dense(cholmod_sparse* input)
{
	return new cholmod_dense(*input);
}

inline cholmod_sparse* cholmod_l_copy_sparse(cholmod_sparse* input)
{
	return new cholmod_sparse(*input);
}



inline SparseMatrix::SparseMatrix(size_t m, size_t n, size_t nnz):
L(*this)
{
    data = cholmod_l_spzeros(m, n, nnz);
}

inline SparseMatrix::SparseMatrix(Triplet& T):
L(*this)
{
    cholmod_triplet *triplet = T.toCholmod();
    data = cholmod_l_triplet_to_sparse(triplet, T.M(), T.N()/*, triplet->nnz*/);
}

inline SparseMatrix::SparseMatrix(cholmod_sparse *data_):
L(*this),
data(data_)
{

}

inline SparseMatrix::SparseMatrix(const SparseMatrix& B):
L(*this),
data(B.copy())
{

}

inline SparseMatrix& SparseMatrix::operator=(cholmod_sparse *data_)
{
    if (data != data_) {
        L.clear();

        clear();
        data = data_;
    }

    return *this;
}

inline SparseMatrix& SparseMatrix::operator=(const SparseMatrix& B)
{
    if (this != &B) {
        L.clear();

        clear();
        data = B.copy();
    }

    return *this;
}

inline SparseMatrix::~SparseMatrix()
{
    clear();
}

inline SparseMatrix SparseMatrix::identity(size_t m, size_t n)
{
    return SparseMatrix(cholmod_l_speye(m, n));
}

inline SparseMatrix SparseMatrix::diag(const DenseMatrix& d)
{
    Triplet T(d.nRows(), d.nRows());
    for (size_t i = 0; i < d.nRows(); i++) T.add(i, i, d(i));

    return SparseMatrix(T);
}

inline SparseMatrix SparseMatrix::transpose() const
{
    return SparseMatrix(cholmod_l_transpose(data/*, 1, common*/));
}

inline size_t SparseMatrix::nRows() const
{
    return data->rows();
}

inline size_t SparseMatrix::nCols() const
{
    return data->cols();
}

inline size_t SparseMatrix::nnz() const
{
	return data->nonZeros(); // cholmod_l_nnz(data, common);
}

inline double SparseMatrix::norm(int norm) const
{
	return data->norm(); // cholmod_l_norm_sparse(data, norm, common);
}

inline SparseMatrix SparseMatrix::submatrix(size_t r0, size_t r1, size_t c0, size_t c1) const
{
	return SparseMatrix(new cholmod_sparse(data->block(r0, c0, r1-r0, c1-c0)));

	
	//cholmod_triplet triplets;
	//for (int k = 0; k < r.size(); ++k)
	//{
	//	int indexer = 0;
	//	for (Eigen::SparseMatrix<double>::InnerIterator it(*data, r[k]); it; ++it)
	//	{
	//		const int col = it.col();
	//		while (indexer < c.size() && col > c[indexer])
	//			++indexer;

	//		if (indexer >= c.size())
	//			break;

	//		if (col == c[indexer]) {
	//			triplets.emplace_back(k, indexer, it.value());
	//			++indexer;
	//		}
	//	}
	//}

	//return cholmod_l_triplet_to_sparse(&triplets, r.size(), c.size());



    /*SuiteSparse_long rsize = (SuiteSparse_long)(r1 - r0);
    SuiteSparse_long *rset = new SuiteSparse_long[rsize];
    for (size_t i = 0; i < rsize; i++) rset[i] = r0 + i;

    SuiteSparse_long csize = (SuiteSparse_long)(c1 - c0);
    SuiteSparse_long *cset = new SuiteSparse_long[csize];
    for (size_t j = 0; j < csize; j++) cset[j] = c0 + j;

    data->stype = 0;
    SparseMatrix A(cholmod_l_submatrix(data, rset, rsize, cset, csize, 1, 1, common));
    delete[] rset;
    delete[] cset;

    return A;*/
}

inline SparseMatrix SparseMatrix::submatrix(const std::vector<int>& rows, const std::vector<int>& cols) const
{
	//auto d = *data;
	//return SparseMatrix(new cholmod_sparse(d(Eigen::placeholders::all, r, c)));

	//vector<int> c(rows);
	//std::sort(c.begin(), c.end());

	cholmod_triplet triplets;
	for (int c = 0; c < cols.size(); ++c)
	{
		int indexer = 0;
		for (Eigen::SparseMatrix<double>::InnerIterator it(*data, cols[c]); it; ++it)
		{
			const int row = it.row();
			//std::cout << col;
			while (indexer < rows.size() && row > rows[indexer])
				++indexer;

			if (indexer >= rows.size())
				break;

			if (row == rows[indexer]) {
				triplets.emplace_back(indexer, c, it.value());
				++indexer;
			}
		}
	}

	return cholmod_l_triplet_to_sparse(&triplets, rows.size(), cols.size());
}

//inline SparseMatrix SparseMatrix::submatrix(const vector<int>& copy, const vector<int>& r) const
//{
//	//auto d = *data;
//	//return SparseMatrix(new cholmod_sparse(d(Eigen::placeholders::all, r, c)));
//
//	vector<int> c(copy);
//	std::sort(c.begin(), c.end());
//
//	cholmod_triplet triplets;
//	for (int k = 0; k < r.size(); ++k)
//	{
//		int indexer = 0;
//		for (Eigen::SparseMatrix<double>::InnerIterator it(*data, r[k]); it; ++it)
//		{
//			const int col = it.row();
//			//std::cout << col;
//			while (indexer < c.size() && col > c[indexer])
//				++indexer;
//
//			if (indexer >= c.size())
//				break;
//
//			if (col == c[indexer]) {
//				triplets.emplace_back(indexer, k, it.value());
//				++indexer;
//			}	
//		}
//	}
//
//	return cholmod_l_triplet_to_sparse(&triplets, c.size(), r.size());
//}

inline DenseMatrix SparseMatrix::toDense() const
{
    return DenseMatrix(cholmod_l_sparse_to_dense(data));
}

inline cholmod_sparse* SparseMatrix::copy() const
{
    return cholmod_l_copy_sparse(data);
}

inline cholmod_sparse* SparseMatrix::toCholmod()
{
    return data;
}

//inline void scale(double s, cholmod_sparse *A)
//{
//    // A = s*A
//    DenseMatrix S(1, 1);
//    S(0, 0) = s;
//    cholmod_l_scale(S.toCholmod(), CHOLMOD_SCALAR, A, common);
//}
//
//inline cholmod_sparse* add(cholmod_sparse *A, cholmod_sparse *B, double alpha[2], double beta[2])
//{
//    // C = alpha*A + beta*B
//    return cholmod_l_add(A, B, alpha, beta, 1, 1, common);
//}
//
//inline cholmod_sparse* mul(cholmod_sparse *A, cholmod_sparse *B)
//{
//    // C = A*B
//    return cholmod_l_ssmult(A, B, 0, 1, 1, common);
//}
//
//inline void mul(cholmod_sparse *A, cholmod_dense *X, cholmod_dense *Y, double alpha[2], double beta[2])
//{
//    // Y = alpha*(A*X) + beta*Y
//    cholmod_l_sdmult(A, 0, alpha, beta, X, Y, common);
//}

inline SparseMatrix operator*(const SparseMatrix& A, double s)
{
	return SparseMatrix(new cholmod_sparse((*A.data) * s));
    /*cholmod_sparse *data = A.copy();
    scale(s, data);

    return SparseMatrix(data);*/
}

inline SparseMatrix operator+(const SparseMatrix& A, const SparseMatrix& B)
{
	return SparseMatrix(new cholmod_sparse((*A.data) + (*B.data)));
    /*double alpha[2] = {1.0, 1.0};
    double beta[2] = {1.0, 1.0};
    return SparseMatrix(add(A.data, B.data, alpha, beta));*/
}

inline SparseMatrix operator-(const SparseMatrix& A, const SparseMatrix& B)
{
	return SparseMatrix(new cholmod_sparse((*A.data) - (*B.data)));
    /*double alpha[2] = {1.0, 1.0};
    double beta[2] = {-1.0, -1.0};
    return SparseMatrix(add(A.data, B.data, alpha, beta));*/
}

inline SparseMatrix operator*(const SparseMatrix& A, const SparseMatrix& B)
{
	return SparseMatrix(new cholmod_sparse((*A.data) * (*B.data)));
    //return SparseMatrix(mul(A.data, B.data));
}

inline DenseMatrix operator*(const SparseMatrix& A, const DenseMatrix& X)
{
	return DenseMatrix(new cholmod_dense((*A.data) * (*X.data)));
    /*DenseMatrix Y(A.nRows(), X.nCols());
    double alpha[2] = {1.0, 1.0};
    double beta[2] = {0.0, 0.0};
    mul(A.data, X.data, Y.data, alpha, beta);

    return Y;*/
}

inline SparseMatrix& operator*=(SparseMatrix& A, double s)
{
	return SparseMatrix(new cholmod_sparse((*A.data) * s));
    /*scale(s, A.data);
    A.L.clearNumeric();

    return A;*/
}

inline SparseMatrix& operator+=(SparseMatrix& A, const SparseMatrix& B)
{
	return SparseMatrix(new cholmod_sparse((*A.data) + (*B.data)));
    /*double alpha[2] = {1.0, 1.0};
    double beta[2] = {1.0, 1.0};
    A = add(A.data, B.data, alpha, beta);

    return A;*/
}

inline SparseMatrix& operator-=(SparseMatrix& A, const SparseMatrix& B)
{
	return SparseMatrix(new cholmod_sparse((*A.data) - (*B.data)));
    /*double alpha[2] = {1.0, 1.0};
    double beta[2] = {-1.0, -1.0};
    A = add(A.data, B.data, alpha, beta);

    return A;*/
}

inline void SparseMatrix::clear()
{
	delete data;
    //cholmod_l_free_sparse(&data, common);
    data = NULL;
}

inline Triplet::Triplet(size_t m_, size_t n_):
m(m_),
n(n_),
capacity(m_)
{
	data = new cholmod_triplet(); // cholmod_l_allocate_triplet(m, n, capacity, 0, CHOLMOD_REAL, common);
    //data->nnz = 0;
}

inline Triplet::~Triplet()
{
    clear();
}

inline void Triplet::add(size_t i, size_t j, double x)
{
    /*if (data->nnz == capacity) increaseCapacity();

    ((size_t *)data->i)[data->nnz] = i;
    ((size_t *)data->j)[data->nnz] = j;
    ((double *)data->x)[data->nnz] = x;
    data->nnz++;*/
	data->emplace_back(i, j, x);
}

inline cholmod_triplet* Triplet::toCholmod()
{
    return data;
}

inline void Triplet::increaseCapacity()
{
    //// create triplet with increased capacity
    //capacity *= 2;
    //cholmod_triplet *newData = cholmod_l_allocate_triplet(m, n, capacity, 0, CHOLMOD_REAL, common);
    //memcpy(newData->i, data->i, data->nzmax*sizeof(size_t));
    //memcpy(newData->j, data->j, data->nzmax*sizeof(size_t));
    //memcpy(newData->x, data->x, data->nzmax*sizeof(double));
    //newData->nnz = data->nnz;

    //// clear old triplet and assign the newly created triplet to it
    //clear();
    //data = newData;
}

inline void Triplet::clear()
{
	delete data;
    //cholmod_l_free_triplet(&data, common);
    data = NULL;
}
