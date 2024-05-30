#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <omp.h>
#include <mpi.h>


using namespace std;


int main(int argc, char* argv[]){

    int N_rank, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &N_rank );
    //printf( "Hello World on rank %d/%d\n", rank, N_rank );

    ifstream file_in;
    double terminate=1e-12, L=1.0;

    file_in.open(argv[1]);
    if(argc>=3){
        terminate=atof(argv[2]);
        L=atof(argv[3]);
    }

    size_t N_grid_in, N_in;

    if (file_in.fail()){
        cerr<<"Failed to open file"<<endl;
        return 2;
    }
    else{
        N_grid_in=distance(istream_iterator<double>(file_in), istream_iterator<double>{});
        N_in = ((int) sqrt(N_grid_in));
    }

    file_in.clear();
    file_in.seekg(0);

    if (N_grid_in/N_in != N_in || N_grid_in%N_in != 0){
        cerr<<"Not a square matrix"<<endl;
        return 3;
    }


    const int N=N_in-2, N_grid=N_grid_in;
    double u[N_grid]={0}, source[N_grid]={0};

    for (int i=0; i<N_grid; i++){
        file_in >> source[i];
        //printf("%6e\n", source[i]);
    }
    file_in.close();

    for(int i=1; i<N+1; i++){
        u[i] = source[i];
        u[(N+1)*(N+2)+i] = source[(N+1)*(N+2)+i];
        u[(i)*(N+2)] = source[(i)*(N+2)];
        u[(i)*(N+2)+N+1] = source[(i)*(N+2)+N+1];
        //printf("%6e\n", x[i]);
    }

    double err, err_max;
    int iters=0;

    //double Ad[N_grid] {0}, residual[N_grid] {0}, d[N_grid] {0};

    // MPI thingy
    if ( rank < N_rank - 1 ) {
        const unsigned int N_row=N/N_rank;
        double u_MPI[(N_row+2)*(N+2)] {0}, source_MPI[N_row*(N+2)] {0};
        double boundary_u[N+2] {0}, boundary_d[N+2] {0};
        double Ad[N_row*(N+2)] {0}, residual[N_row*(N+2)] {0}, d[N_row*(N+2)] {0};

        for(int i=0; i<N_row+2; i++){
            for(int j=0; j<N+2; j++){
                u_MPI[i*(N+2)+j]=u[(rank*N_row+i)*(N+2)+j];
                if (i==0){
                    boundary_u[j]=u_MPI[(i)*(N+2)+j];
                }
                if (i==N_row+1){
                    boundary_d[j]=u_MPI[(i)*(N+2)+j];
                }
            }
        }
        for(int i=0; i<N_row; i++){
            for(int j=1; j<N+1; j++){
                source_MPI[i*(N+2)+j]=source[(rank*N_row+i+1)*(N+2)+j];
            }
        }

        // initial residual = b-Ax
        # pragma omp parallel for collapse(2)
        for(int i=1; i<N_row+1; i++){
            for(int j=1; j<N+1; j++){
                residual[i*(N+2)+j] = source_MPI[i*(N+2)+j]
                                      -(u_MPI[(i+1)*(N+2)+j]
                                        +u_MPI[(i-1)*(N+2)+j]
                                        +u_MPI[i*(N+2)+j+1]
                                        +u_MPI[i*(N+2)+j-1]
                                        -4*u_MPI[i*(N+2)+j])*N*N/L/L;
                d[i*(N+2)+j] = residual[i*(N+2)+j];
            }
        }
        do{
            double alpha=0, alpha_num=0, alpha_den=0;
            double beta=0, beta_num=0;
            err=0;
            err_max=0;

            # pragma omp parallel
            {
            // compute Ad = A(d)
            # pragma omp for collapse(2)
            for(int i=1; i<N_row+1; i++){
                for(int j=1; j<N+1; j++){
                    Ad[i*(N+2)+j] = (d[(i+1)*(N+2)+j]
                                     +d[(i-1)*(N+2)+j]
                                     +d[i*(N+2)+j+1]
                                     +d[i*(N+2)+j-1]
                                     -4*d[i*(N+2)+j])*N*N/L/L;
                }
            }
            // alpha = r*r/d*Ad
            # pragma omp for collapse(2) reduction(+:alpha_num, alpha_den)
            for(int i=1; i<N_row+1; i++){
                for(int j=1; j<N+1; j++){
                    alpha_num += residual[i*(N+2)+j]*residual[i*(N+2)+j];
                    alpha_den += d[i*(N+2)+j]*Ad[i*(N+2)+j];
                }
            }
            # pragma omp single
            alpha = alpha_den!=0 ? alpha_num / alpha_den : 0;


            # pragma omp for collapse(2) reduction(+:beta_num)
            for(int i=1; i<N_row+1; i++){
                for(int j=1; j<N+1; j++){
                    u_MPI[i*(N+2)+j] += alpha*d[i*(N+2)+j];
                    residual[i*(N+2)+j] -= alpha*Ad[i*(N+2)+j];
                    beta_num += residual[i*(N+2)+j]*residual[i*(N+2)+j];
                }
            }
            # pragma omp single
            beta = beta_num/alpha_num;

            # pragma omp for collapse(2) reduction(+:err) reduction(max:err_max)
            for(int i=1; i<N_row+1; i++){
                for(int j=1; j<N+1; j++){
                    d[i*(N+2)+j] = residual[i*(N+2)+j] + beta*d[i*(N+2)+j];
                    err += fabs(residual[i*(N+2)+j]);
                    err_max = fabs(residual[i*(N+2)+j]) > err_max ? fabs(residual[i*(N+2)+j]) : err_max;
                }
            }
            }
            err /= N*N;
            iters++;

        }while(err>terminate);
    }

    else{
        const unsigned int N_row=N%N_rank;
        double u_MPI[(N_row+2)*(N+2)] {0}, source_MPI[N_row*(N+2)] {0};
        double boundary_u[N+2] {0}, boundary_d[N+2] {0};
        double Ad[N_row*(N+2)] {0}, residual[N_row*(N+2)] {0}, d[N_row*(N+2)] {0};

        for(int i=0; i<N_row+2; i++){
            for(int j=0; j<N+2; j++){
                u_MPI[i*(N+2)+j]=u[(rank*N_row+i)*(N+2)+j];
                if (i==0){
                    boundary_d[j]=u_MPI[(i)*(N+2)+j];
                }
                if (i==N_row+1){
                    boundary_u[j]=u_MPI[(i)*(N+2)+j];
                }
            }
        }
        for(int i=0; i<N_row; i++){
            for(int j=1; j<N+1; j++){
                source_MPI[i*(N+2)+j]=source[(rank*N_row+i+1)*(N+2)+j];
            }
        }

        // initial residual = b-Ax
        # pragma omp parallel for collapse(2)
        for(int i=1; i<N_row+1; i++){
            for(int j=1; j<N+1; j++){
                residual[i*(N+2)+j] = source_MPI[i*(N+2)+j]
                                      -(u_MPI[(i+1)*(N+2)+j]
                                        +u_MPI[(i-1)*(N+2)+j]
                                        +u_MPI[i*(N+2)+j+1]
                                        +u_MPI[i*(N+2)+j-1]
                                        -4*u_MPI[i*(N+2)+j])*N*N/L/L;
                d[i*(N+2)+j] = residual[i*(N+2)+j];
            }
        }
        do{
            double alpha=0, alpha_num=0, alpha_den=0;
            double beta=0, beta_num=0;
            err=0;
            err_max=0;

            # pragma omp parallel
            {
            // compute Ad = A(d)
            # pragma omp for collapse(2)
            for(int i=1; i<N_row+1; i++){
                for(int j=1; j<N+1; j++){
                    Ad[i*(N+2)+j] = (d[(i+1)*(N+2)+j]
                                     +d[(i-1)*(N+2)+j]
                                     +d[i*(N+2)+j+1]
                                     +d[i*(N+2)+j-1]
                                     -4*d[i*(N+2)+j])*N*N/L/L;
                }
            }
            // alpha = r*r/d*Ad
            # pragma omp for collapse(2) reduction(+:alpha_num, alpha_den)
            for(int i=1; i<N_row+1; i++){
                for(int j=1; j<N+1; j++){
                    alpha_num += residual[i*(N+2)+j]*residual[i*(N+2)+j];
                    alpha_den += d[i*(N+2)+j]*Ad[i*(N+2)+j];
                }
            }
            # pragma omp single
            alpha = alpha_den!=0 ? alpha_num / alpha_den : 0;


            # pragma omp for collapse(2) reduction(+:beta_num)
            for(int i=1; i<N_row+1; i++){
                for(int j=1; j<N+1; j++){
                    u_MPI[i*(N+2)+j] += alpha*d[i*(N+2)+j];
                    residual[i*(N+2)+j] -= alpha*Ad[i*(N+2)+j];
                    beta_num += residual[i*(N+2)+j]*residual[i*(N+2)+j];
                }
            }
            # pragma omp single
            beta = beta_num/alpha_num;

            # pragma omp for collapse(2) reduction(+:err) reduction(max:err_max)
            for(int i=1; i<N_row+1; i++){
                for(int j=1; j<N+1; j++){
                    d[i*(N+2)+j] = residual[i*(N+2)+j] + beta*d[i*(N+2)+j];
                    err += fabs(residual[i*(N+2)+j]);
                    err_max = fabs(residual[i*(N+2)+j]) > err_max ? fabs(residual[i*(N+2)+j]) : err_max;
                }
            }
            }
            err /= N*N;
            iters++;

        }while(err>terminate);

    }

    /*
    // initial residual = b-Ax
    # pragma omp parallel for collapse(2)
    for(int i=1; i<N+1; i++){
        for(int j=1; j<N+1; j++){
            residual[i*(N+2)+j] = source[i*(N+2)+j]
                                  -(u[(i+1)*(N+2)+j]
                                    +u[(i-1)*(N+2)+j]
                                    +u[i*(N+2)+j+1]
                                    +u[i*(N+2)+j-1]
                                    -4*u[i*(N+2)+j])*N*N/L/L;
            d[i*(N+2)+j] = residual[i*(N+2)+j];
        }
    }


    do{
        double alpha=0, alpha_num=0, alpha_den=0;
        double beta=0, beta_num=0;
        err=0;
        err_max=0;

        # pragma omp parallel
        {
        // compute Ad = A(d)
        # pragma omp for collapse(2)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                Ad[i*(N+2)+j] = (d[(i+1)*(N+2)+j]
                                 +d[(i-1)*(N+2)+j]
                                 +d[i*(N+2)+j+1]
                                 +d[i*(N+2)+j-1]
                                 -4*d[i*(N+2)+j])*N*N/L/L;
            }
        }
        // alpha = r*r/d*Ad
        # pragma omp for collapse(2) reduction(+:alpha_num, alpha_den)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                alpha_num += residual[i*(N+2)+j]*residual[i*(N+2)+j];
                alpha_den += d[i*(N+2)+j]*Ad[i*(N+2)+j];
            }
        }
        # pragma omp single
        alpha = alpha_den!=0 ? alpha_num / alpha_den : 0;


        # pragma omp for collapse(2) reduction(+:beta_num)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                u[i*(N+2)+j] += alpha*d[i*(N+2)+j];
                residual[i*(N+2)+j] -= alpha*Ad[i*(N+2)+j];
                beta_num += residual[i*(N+2)+j]*residual[i*(N+2)+j];
            }
        }
        # pragma omp single
        beta = beta_num/alpha_num;

        # pragma omp for collapse(2) reduction(+:err) reduction(max:err_max)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                d[i*(N+2)+j] = residual[i*(N+2)+j] + beta*d[i*(N+2)+j];
                err += fabs(residual[i*(N+2)+j]);
                err_max = fabs(residual[i*(N+2)+j]) > err_max ? fabs(residual[i*(N+2)+j]) : err_max;
            }
        }
        }
        err /= N*N;
        iters++;

    }while(err>terminate);
    */


    
    printf("%d iterations, avg residual=%6e, max residual=%6e\n", iters, err, err_max);

    FILE *file_out;
    std::string buf("output_");
    buf.append(std::to_string(N));
    buf.append(".txt");
    file_out = fopen(buf.data(), "w");
    for(int i=0; i<N_grid; i++){
        //printf("%.18f ", d[i]);
        fprintf(file_out, "%.18f ", u[i]);
    }
    fclose(file_out);

    MPI_Finalize();
    return 0;
}
