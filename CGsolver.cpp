#include <iostream>
#include <cmath>
#include <vector>
#include <limits> // For numeric_limits

using namespace std;

using vec = vector<double>; // vector
using matrix = vector<vec>; // matrix (=collection of (row) vectors)

// vector addition and subtraction
vec vecAddSub(const vec &x, const vec &y, double alpha)
{
  int N = x.size();
  vec z(N);
  for (int i = 0; i < N; i++)
  {
    z[i] = x[i] + alpha * y[i];
  }
  return z;
}

// Matrix-vector multiplication
vec matvec(const matrix &A, const vec &x)
{
  int N = x.size();
  vec y(N, 0.0);
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      y[i] += A[i][j] * x[j];
    }
  }
  return y;
}

// Dot product
double dot(const vec &x, const vec &y)
{
  int N = x.size();
  double result = 0.0;
  for (int i = 0; i < N; i++)
  {
    result += x[i] * y[i];
  }
  return result;
}

// Conjungate Gradient Method
vec CG(const matrix &A, const vec &b, vec &x, const double tol)
{
  // Initialize variables
  int N = b.size();
  vec r = vecAddSub(b, matvec(A, x), -1.0); // initial residue
  vec p = r;                                // initial search direction
  double alpha, beta, r_dot, r_dot_new;     // variables for CG

  // Iterate
  for (int k = 0; k < N; k++)
  {
    r_dot = dot(r, r);
    alpha = r_dot / dot(p, matvec(A, p));
    x = vecAddSub(x, p, alpha);
    r = vecAddSub(r, matvec(A, p), -alpha);
    r_dot_new = dot(r, r);
    if (sqrt(r_dot_new) < tol)
    {
      break;
    }
    beta = r_dot_new / r_dot;
    p = vecAddSub(r, p, beta);
  }

  return x;
}

int main()
{
  // Define the matrix A and the vector b
  matrix A = {{4, -1, 0, -1, 0, 0},
              {-1, 4, -1, 0, -1, 0},
              {0, -1, 4, 0, 0, -1},
              {-1, 0, 0, 4, -1, 0},
              {0, -1, 0, -1, 4, -1},
              {0, 0, -1, 0, -1, 4}};
  vec b = {0, 0, 0, 1, 1, 1};

  // Initial guess of the solution
  vec x = {0, 0, 0, 0, 0, 0};

  // Tolerance
  double tol = 1.0e-6;

  // Solve the linear system using the Conjugate Gradient Method
  CG(A, b, x, tol);

  // Output the solution
  for (int i = 0; i < x.size(); i++)
  {
    cout << "x[" << i << "] = " << x[i] << endl;
  }

  return 0;
}