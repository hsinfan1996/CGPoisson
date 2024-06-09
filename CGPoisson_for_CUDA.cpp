#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

using vec = vector<long double>; // vector
using matrix = vector<vec>;      // matrix (=collection of (row) vectors)

// vector addition (w/ and w/o using Kahan Summation Algorithm)
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

vec vecAddSub_Kahan(const vec &x, const vec &y, double alpha)
{
  int N = x.size();
  vec z(N);
  std::vector<long double> c(N, 0.0); // Compensation vector for Kahan summation

  for (int i = 0; i < N; i++)
  {
    long double y_scaled = alpha * static_cast<long double>(y[i]);
    long double temp = x[i] - c[i];
    long double sum = static_cast<long double>(temp) + y_scaled;
    c[i] = (sum - temp) - y_scaled;
    z[i] = sum;
  }
  return z;
}

// Dot product (w/ and w/o using Kahan Summation Algorithm)
long double dot(const vec &x, const vec &y)
{
  int N = x.size();
  long double result = 0.0;
  for (int i = 0; i < N; i++)
  {
    result += static_cast<long double>(x[i]) * static_cast<long double>(y[i]);
  }
  return result;
}

long double dot_Kahan(const vec &x, const vec &y)
{
  int N = x.size();
  long double sum = 0.0;
  long double c = 0.0; // A running compensation for lost low-order bits.

  for (int i = 0; i < N; i++)
  {
    long double product = static_cast<long double>(x[i]) * static_cast<long double>(y[i]);
    long double y = product - c; // So far, so good: c is zero.
    long double t = sum + y;     // Alas, sum is big, y small, so low-order digits of y are lost.
    c = (t - sum) - y;           // (t - sum) cancels the high-order part of y; subtracting y recovers negative (low part of y)
    sum = t;                     // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
  }

  return sum;
}

// Define the operation of the discretized laplace operator
vec poissonAx(const vec x, const int &N_in)
{
  vec Ax(N_in * N_in, 0.0);
  for (int j = 0; j < N_in; j++)
  {
    for (int i = 0; i < N_in; i++)
    {
      int idx = i + j * N_in;
      if (i == 0 && j == 0)
      {
        Ax[idx] = 4.0 * x[idx] - x[idx + 1] - x[idx + N_in];
      }
      else if (i == 0 && j == N_in - 1)
      {
        Ax[idx] = 4.0 * x[idx] - x[idx + 1] - x[idx - N_in];
      }
      else if (i == 0)
      {
        Ax[idx] = 4.0 * x[idx] - x[idx + 1] - x[idx + N_in] - x[idx - N_in];
      }
      else if (i == N_in - 1 && j == 0)
      {
        Ax[idx] = 4.0 * x[idx] - x[idx - 1] - x[idx + N_in];
      }
      else if (i == N_in - 1 && j == N_in - 1)
      {
        Ax[idx] = 4.0 * x[idx] - x[idx - 1] - x[idx - N_in];
      }
      else if (i == N_in - 1)
      {
        Ax[idx] = 4.0 * x[idx] - x[idx - 1] - x[idx + N_in] - x[idx - N_in];
      }
      else if (j == 0)
      {
        Ax[idx] = 4.0 * x[idx] - x[idx - 1] - x[idx + 1] - x[idx + N_in];
      }
      else if (j == N_in - 1)
      {
        Ax[idx] = 4.0 * x[idx] - x[idx - 1] - x[idx + 1] - x[idx - N_in];
      }
      else
      {
        Ax[idx] = 4.0 * x[idx] - x[idx - 1] - x[idx + 1] - x[idx + N_in] - x[idx - N_in];
      }
    }
  }

  return Ax;
}

// Conjungate Gradient Method
vec CG(const vec &b, const vec &x_0, int &N_in, const double tol)
{
  // Initialize variables
  vec x = x_0; // initial guess of the solution
  vec Ax_0 = poissonAx(x_0, N_in);
  vec r = vecAddSub(b, Ax_0, -1.0);   // initial residue
  vec p = r;                          // initial search direction
  long double alpha, beta, r_dot_new; // variables for CG
  long double r_dot, b_dot, Ax_dot;

  // Iterate
  // double x_i, x_f;
  for (int k = 0; k < N_in; k++)
  {
    r_dot = dot(r, r);
    cout << "r_dot[" << k << "] = " << r_dot << endl;
    alpha = r_dot / dot(p, poissonAx(p, N_in));
    // cout << "alpha = " << alpha << endl;
    // cout << "p[645] = " << p[645] << endl;
    // x_i = x[645];
    x = vecAddSub(x, p, alpha);
    // x_f = x[645];
    // cout << "delta x[645] = " << x_f - x_i << endl;
    r = vecAddSub(r, poissonAx(p, N_in), -alpha);
    r_dot_new = dot(r, r);
    cout << "r_dot[" << k + 1 << "] = " << r_dot_new << endl;
    if (r_dot_new < tol)
    {
      break;
    }
    beta = r_dot_new / r_dot;
    p = vecAddSub(r, p, beta);
  }
  // check the error of the solution
  Ax_0 = poissonAx(x, N_in);
  r = vecAddSub(b, Ax_0, -1.0);
  cout << "Error = " << dot(r, r) << endl;

  return x;
}

// read the source term from file and construct the right-hand side of the linear system
void readInput(const string &filename, int &N, vec &BC, vec &b, double &delta)
{
  ifstream infile(filename);
  if (!infile)
  {
    cerr << "Error: file " << filename << " not found." << endl;
    exit(1);
  }

  infile >> N;
  vec rho(N * N, 0.0);
  for (int i = 0; i < N * N; i++)
  {
    infile >> rho[i];
  }
  BC.resize(4 * N); // BC: boundary conditions (x=0, x=1, y=0, y=1)
  for (int i = 0; i < 4 * N; i++)
  {
    infile >> BC[i];
  }

  int N_in = N - 2;
  b.resize(N_in * N_in);
  int index, index_in;
  for (int i = 0; i < N_in; i++)
  {
    for (int j = 0; j < N_in; j++)
    {
      index_in = i + j * N_in;
      index = i + 1 + (j + 1) * N;
      if (i == 0 && j == 0)
      {
        b[index_in] = -rho[index] * delta * delta + BC[1] + BC[2 * N + 1];
      }
      else if (i == 0 && j == N_in - 1)
      {
        b[index_in] = -rho[index] * delta * delta + BC[N - 2] + BC[3 * N + 1];
      }
      else if (i == N_in - 1 && j == 0)
      {
        b[index_in] = -rho[index] * delta * delta + BC[N + 1] + BC[3 * N - 2];
      }
      else if (i == N_in - 1 && j == N_in - 1)
      {
        b[index_in] = -rho[index] * delta * delta + BC[2 * N - 2] + BC[4 * N - 2];
      }
      else if (i == 0)
      {
        b[index_in] = -rho[index] * delta * delta + BC[j + 1];
      }
      else if (i == N_in - 1)
      {
        b[index_in] = -rho[index] * delta * delta + BC[j + N + 1];
      }
      else if (j == 0)
      {
        b[index_in] = -rho[index] * delta * delta + BC[i + N * 2 + 1];
      }
      else if (j == N_in - 1)
      {
        b[index_in] = -rho[index] * delta * delta + BC[i + N * 3 + 1];
      }
      else
      {
        b[index_in] = -rho[index] * delta * delta;
      }
    }
  }

  infile.close();
}

int main()
{
  // parameters
  int N, N_in;        // N is the number of grid points in one direction, N_in is the number of grid points in one direction excluding the boundary points
  double delta = 1.0; // delta is the grid spacing
  vec b;

  // Read input from file
  vec BC;
  cout << "Type the input file and output file: " << endl;
  string inputfile, outputfile;
  cin >> inputfile, outputfile;
  readInput(inputfile, N, BC, b, delta);

  N_in = N - 2;
  vec x(N_in * N_in, 0.0);   // the solution vector x (the solution excluding the boundary points)
  vec x_0(N_in * N_in, 0.0); // Initial guess of the solution
  cout << "loading b and x_0 success" << endl;

  /* Generate the source term b and save it to a file for testing CUDA code
  string b_file = "b_16.txt";
  ofstream outFile_b("b_16.txt");
  if (!outFile_b)
  {
    cerr << "Error opening file for writing: " << outputfile << endl;
  }

  for (double val : b)
  {
    outFile_b << val << ", ";
  }
  outFile_b.close();
  */

  // Tolerance
  double tol = 1.0e-6;

  // Solve the linear system using the Conjugate Gradient Method
  x = CG(b, x_0, N_in, tol);

  cerr << "CG solver Success " << endl;

  // reshape the solution vector x to a matrix x_mat and include the bourndary
  matrix x_mat(N, vec(N, 0.0));
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      if (i == 0)
      {
        x_mat[i][j] = BC[j];
      }
      else if (i == N - 1)
      {
        x_mat[i][j] = BC[j + N];
      }
      else if (j == 0)
      {
        x_mat[i][j] = BC[i + 2 * N];
      }
      else if (j == N - 1)
      {
        x_mat[i][j] = BC[i + 3 * N];
      }
      else
      {
        x_mat[i][j] = x[(i - 1) + (j - 1) * N_in]; // / (64 - 1) / (64 - 1);
      }
    }
  }
  cerr << "X reshape Success " << endl;

  // save the solution phi into a file
  ofstream outFile(outputfile);
  if (!outFile)
  {
    cerr << "Error opening file for writing: " << outputfile << endl;
  }

  for (const auto &row : x_mat)
  {
    for (double val : row)
    {
      outFile << val << " ";
    }
    outFile << "\n";
  }

  outFile.close();

  cerr << "Save the solution x in" << outputfile << endl;

  return 0;
}