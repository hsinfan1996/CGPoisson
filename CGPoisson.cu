#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

using vec = vector<long double>; // vector
using matrix = vector<vec>;      // matrix (=collection of (row) vectors)

// Matrix-vector multiplication
__global__ void matVec(double *A, double *x, double *result, int N)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N)
  {
    double sum = 0.0;
    for (int col = 0; col < N; ++col)
    {
      sum += A[row * N + col] * x[col];
    }
    result[row] = sum;
  }
}

// vector addition and subtraction (for arbitrary long vector)
__global__ void vecAddSub(double *a, double *b, double *result, double alpha, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < N)
  {
    result[tid] = a[tid] + alpha * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  __syncthreads();
}

// Dot product
__global__ void dot(double *a, double *b, double *result, int N)
{
  extern __shared__ double cache[];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIdx = threadIdx.x;

  double temp = 0.0;
  while (tid < N)
  {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  cache[cacheIdx] = temp;

  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0)
  {
    if (cacheIdx < i)
    {
      cache[cacheIdx] += cache[cacheIdx + i];
    }
    __syncthreads();
    i /= 2;
  }

  if (cacheIdx == 0)
  {
    result[blockIdx.x] = cache[0];
  }
}

// Host-side code to accumulate the dot results
void host_dot(double *a, double *b, double *result, int N, int blocksPerGrid, int threadsPerBlock)
{
  double *d_block_results;
  cudaMalloc(&d_block_results, blocksPerGrid * sizeof(double));
  cudaMemset(d_block_results, 0, blocksPerGrid * sizeof(double));

  dot<<<blocksPerGrid, threadsPerBlock>>>(a, b, d_block_results, N);

  // Copy partial results back to the host
  double *h_block_results = new double[blocksPerGrid];
  cudaMemcpy(h_block_results, d_block_results, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

  // Sum the partial results on the host
  double finalResult = 0.0;
  for (int i = 0; i < blocksPerGrid; ++i)
  {
    finalResult += h_block_results[i];
  }

  *result = finalResult;

  delete[] h_block_results;
  cudaFree(d_block_results);
}

// Define the operation of the discretized laplace operator
__global__ void poissonAx(double *u, double *result, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i + j * N;

  if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
  {
    result[idx] = 4.0 * u[idx] - u[idx + 1] - u[idx - 1] - u[idx + N] - u[idx - N];
  }
  else if (i == 0 && j == 0)
  {
    result[idx] = 4.0 * u[idx] - u[idx + 1] - u[idx + N];
  }
  else if (i == 0 && j == N - 1)
  {
    result[idx] = 4.0 * u[idx] - u[idx + 1] - u[idx - N];
  }
  else if (i == 0)
  {
    result[idx] = 4.0 * u[idx] - u[idx + 1] - u[idx + N] - u[idx - N];
  }
  else if (i == N - 1 && j == 0)
  {
    result[idx] = 4.0 * u[idx] - u[idx - 1] - u[idx + N];
  }
  else if (i == N - 1 && j == N - 1)
  {
    result[idx] = 4.0 * u[idx] - u[idx - 1] - u[idx - N];
  }
  else if (i == N - 1)
  {
    result[idx] = 4.0 * u[idx] - u[idx - 1] - u[idx + N] - u[idx - N];
  }
  else if (j == 0)
  {
    result[idx] = 4.0 * u[idx] - u[idx - 1] - u[idx + 1] - u[idx + N];
  }
  else if (j == N - 1)
  {
    result[idx] = 4.0 * u[idx] - u[idx - 1] - u[idx + 1] - u[idx - N];
  }
}

// Conjungate Gradient Method
void CG(double *b, double *x, int N, double tol = 1e-6, int maxIter = 1000)
{
  // Allocate device memory
  double *d_b, *d_x, *d_r, *d_p, *d_Ap;
  // double *d_A;
  // cudaMalloc(&d_A, N * N * sizeof(double));
  cudaMalloc(&d_b, N * N * sizeof(double));
  cudaMalloc(&d_x, N * N * sizeof(double));
  cudaMalloc(&d_r, N * N * sizeof(double));
  cudaMalloc(&d_p, N * N * sizeof(double));
  cudaMalloc(&d_Ap, N * N * sizeof(double));

  // Copy data from host to device
  // cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, N * N * sizeof(double), cudaMemcpyHostToDevice);

  // choose the number of threads per block and blocks per grid
  int threadsPerBlock = 256;
  int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

  // Initialize r = b - Ax
  // matVec<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_r, N);        // A * x
  poissonAx<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_r, N);              // A * x
  vecAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_r, d_r, -1, N * N); // r = b - Ax
  cudaMemcpy(d_p, d_r, N * N * sizeof(double), cudaMemcpyDeviceToDevice);  // p = r

  double r_dot = 0.0;
  host_dot(d_r, d_r, &r_dot, N * N, blocksPerGrid, threadsPerBlock); // r_dot = r * r
  printf("r_dot = %d\n", r_dot);

  // Iterate
  double alpha = 0.0;
  double r_dot_new = 0.0;
  for (int i = 0; i < 10; ++i)
  {
    // alpha = r_dot / (p * Ap)
    // matVec<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_p, d_Ap, N); // A * p
    poissonAx<<<blocksPerGrid, threadsPerBlock>>>(d_p, d_Ap, N);        // A * p
    host_dot(d_p, d_Ap, &alpha, N * N, blocksPerGrid, threadsPerBlock); // p * Ap
    alpha = r_dot / alpha;

    vecAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_p, d_x, alpha, N * N);   // x = x + alpha * p
    vecAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_Ap, d_r, -alpha, N * N); // r = r - alpha * Ap

    // r_dot_new = r * r
    host_dot(d_r, d_r, &r_dot_new, N * N, blocksPerGrid, threadsPerBlock);
    printf("r_dot_new = %d\n", r_dot_new);

    if (sqrt(r_dot_new) < tol)
    {
      break;
    }

    double beta = r_dot_new / r_dot;
    vecAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_p, d_p, beta, N); // p = r + beta * p
    r_dot = r_dot_new;
  }

  cudaMemcpy(x, d_x, N * N * sizeof(double), cudaMemcpyDeviceToHost); // Copy x to host

  // cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_Ap);
}

void readInput(const string &filename, int &N, double *&BC, double *&b, float &delta)
{
  ifstream infile(filename);
  if (!infile)
  {
    printf("Error: file not found.");
    exit(1);
  }

  infile >> N;
  double *rho = new double[N * N];
  for (int i = 0; i < N * N; i++)
  {
    infile >> rho[i];
  }
  for (int i = 0; i < 5; i++)
  {
    printf("rho = %d", rho[i]);
  }
  for (int i = 0; i < 4 * N; i++)
  {
    infile >> BC[i];
  }
  for (int i = 0; i < 5; i++)
  {
    printf("BC = %d", BC[i]);
  }

  int N_in = N - 2;
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
  int gid; // GPU_ID

  printf("Enter the GPU ID (0/1): ");
  scanf("%d", &gid);
  printf("%d\n", gid);

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  err = cudaSetDevice(gid);
  if (err != cudaSuccess)
  {
    printf("!!! Cannot select GPU with device ID = %d\n", gid);
    exit(1);
  }
  printf("Select GPU with device ID = %d\n", gid);

  cudaSetDevice(gid);

  // parameters
  int N, N_in; // N is the number of grid points in one direction, N_in is the number of grid points in one direction excluding the boundary points
  printf("Enter the size (N, N) of the 2D lattice: ");
  scanf("%d", &N);
  printf("%d\n", N);
  N_in = N - 2;

  double *b = new double[N_in * N_in];
  float delta = 1.0; // delta is the grid spacing
  // Read input from file
  double *BC = new double[4 * N];
  /*
  string inputfile;
  scanf("%99s", &inputfile);
  readInput(inputfile, N, BC, b, delta);
  */
  readInput("source_512.txt", N, BC, b, delta);

  printf("b = %d", b[N * N / 2 + N / 2]);

  double *x = new double[N_in * N_in]; // the solution vector x (the solution excluding the boundary points)

  // Tolerance
  double tol = 1.0e-6;

  // Solve the linear system using the Conjugate Gradient Method
  CG(b, x, N_in, tol);

  printf("CG solver Success\n");

  /*
  printf("Solution x: \n");
  for (int i = 0; i < N; ++i)
  {
    printf("%f ", x[i]);
  }
  */

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
  printf("X reshape Success \n");

  // save the solution phi into a file
  // const string filename = "phi_CPU.txt";
  ofstream outFile("phi_CPU_512.txt");
  /*
  string outputfile;
  scanf("%99s", &outputfile);
  ofstream outFile(outputfile);
  */
  if (!outFile)
  {
    // printf("Error opening file for writing: %s", outputfile);
    printf("Error opening file for writing phi_CPU_512.txt");
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
}
