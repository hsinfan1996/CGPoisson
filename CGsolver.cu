#include <iostream>
#include <cmath>

using namespace std;

/*
// Custom atomicAdd for double precision if not provided by CUDA
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif
*/

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

// vector addition and subtraction
__global__ void vecAddSub(double *a, double *b, double *result, double alpha, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
  {
    result[tid] = a[tid] + alpha * b[tid];
  }
}

// Dot product
__global__ void dot(double *a, double *b, double *result, int N)
{
  __shared__ double cache[256];
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
    // atomicAdd(result, cache[0]); // Atomic add does not work here
  }
}

// Host-side code to accumulate the dot results
void host_dot(double *a, double *b, double *result, int N, int blocksPerGrid, int threadsPerBlock)
{
  double *d_block_results;
  cudaMalloc(&d_block_results, blocksPerGrid * sizeof(double));
  // cudaMemset(d_block_results, 0, blocksPerGrid * sizeof(double));

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

// Conjungate Gradient Method
void CG(double *A, double *b, double *x, int N, double tol = 1e-6, int maxIter = 1000)
{
  // Allocate device memory
  double *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap;
  cudaMalloc(&d_A, N * N * sizeof(double));
  cudaMalloc(&d_b, N * sizeof(double));
  cudaMalloc(&d_x, N * sizeof(double));
  cudaMalloc(&d_r, N * sizeof(double));
  cudaMalloc(&d_p, N * sizeof(double));
  cudaMalloc(&d_Ap, N * sizeof(double));

  // Copy data from host to device
  cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

  // choose the number of threads per block and blocks per grid
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Initialize r = b - Ax
  matVec<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_r, N);        // A * x
  vecAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_r, d_r, -1, N); // r = b - Ax
  cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice);  // p = r

  double r_dot = 0.0;
  host_dot(d_r, d_r, &r_dot, N * N, blocksPerGrid, threadsPerBlock); // r_dot = r * r

  // Iterate
  double alpha = 0.0;
  double r_dot_new = 0.0;
  for (int i = 0; i < maxIter; ++i)
  {
    // alpha = r_dot / (p * Ap)
    matVec<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_p, d_Ap, N);      // A * p
    host_dot(d_p, d_Ap, &alpha, N * N, blocksPerGrid, threadsPerBlock); // p * Ap
    alpha = r_dot / alpha;

    vecAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_p, d_x, alpha, N);   // x = x + alpha * p
    vecAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_Ap, d_r, -alpha, N); // r = r - alpha * Ap

    // r_dot_new = r * r
    host_dot(d_r, d_r, &r_dot_new, N * N, blocksPerGrid, threadsPerBlock);

    if (sqrt(r_dot_new) < tol)
    {
      break;
    }

    double beta = r_dot_new / r_dot;
    vecAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_p, d_p, beta, N); // p = r + beta * p
    r_dot = r_dot_new;
  }

  cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost); // Copy x to host

  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_Ap);
}

int main()
{
  const int N = 6;

  double A[N][N] = {{4, -1, 0, -1, 0, 0},
                    {-1, 4, -1, 0, -1, 0},
                    {0, -1, 4, 0, 0, -1},
                    {-1, 0, 0, 4, -1, 0},
                    {0, -1, 0, -1, 4, -1},
                    {0, 0, -1, 0, -1, 4}};
  double b[N] = {0, 0, 0, 1, 1, 1};
  double x[N] = {0, 0, 0, 0, 0, 0}; // Initial guess

  CG(&A[0][0], b, x, N);

  cout << "Solution x: ";
  for (int i = 0; i < N; ++i)
  {
    cout << x[i] << " ";
  }
  cout << endl;

  return 0;
}
