#include "SparseMatrix.h"
#include <Eigen/Sparse>

//extern Common common;

typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> cholmod_factor;

inline cholmod_factor* cholmod_l_analyze(cholmod_sparse *C)
{
	cholmod_factor* f = new cholmod_factor();
	f->analyzePattern(*C);
	return f;
}

inline bool cholmod_l_factorize(cholmod_sparse *C, cholmod_factor *f)
{
	f->factorize(*C);
	return true;
}

inline DenseMatrix cholmod_l_solve(cholmod_factor *f, cholmod_dense *B)
{
	return DenseMatrix(new cholmod_dense(f->solve(*B)));
}

inline Cholesky::Cholesky(SparseMatrix& A_):
A(A_),
factor(NULL),
validSymbolic(false),
validNumeric(false)
{

}

inline Cholesky::~Cholesky()
{
    clear();
}

inline void Cholesky::clear()
{
    if (factor) {
        //cholmod_l_free_factor(&factor, common);
		delete factor;
        factor = NULL;
    }

    validSymbolic = false;
    validNumeric = false;
}

inline void Cholesky::clearNumeric()
{
    validNumeric = false;
}

inline void Cholesky::buildSymbolic(cholmod_sparse *C)
{
    clear();

    factor = cholmod_l_analyze(C);
    if (factor) validSymbolic = true;
}

inline void Cholesky::buildNumeric(cholmod_sparse *C)
{
    if (factor) validNumeric = (bool)cholmod_l_factorize(C, factor);
}

inline void Cholesky::update()
{
    cholmod_sparse *C = A.toCholmod();
    //C->stype = 1;

    if (!validSymbolic) buildSymbolic(C);
    if (!validNumeric) buildNumeric(C);
}

inline void Cholesky::solvePositiveDefinite(DenseMatrix& x, DenseMatrix& b)
{
    update();
    if (factor) x = cholmod_l_solve(factor, b.toCholmod());
}
