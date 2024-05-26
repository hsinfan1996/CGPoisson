#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <omp.h>


double ref_func(double x, double y){
    return y*(1-y)*pow(x, 3);
}

double source_term(double x, double y){
    return 6*x*y*(1-y)-2*pow(x, 3);
}

int main(int argc, char* argv[]){

    int N_in;
    if(argc>=2) N_in=atof(argv[1]);
    else N_in=100;

    const int N=N_in, N_grid=(N_in+2)*(N_in+2);
    double L=1.0;
    double terminate=1e-12;

    double dx=L/N;
    double x[N+2] {0};
    x[N+1]=L;
    double u[N_grid] {0};
    double source[N_grid] {0}, ref[N_grid] {0};

    for(int i=1; i<N+1; i++){
        x[i] = (i-0.5)*dx;
        //printf("%6e\n", x[i]);
    }

    // boundary condition
    for(int j=1; j<N+1; j++){
        u[(N+1)*(N+2)+j] = x[j]*(1-x[j]);
        //printf("%3e\n", u[N+1][j]);
    }

    for(int i=1; i<N+1; i++){
        for(int j=1; j<N+1; j++){
            source[i*(N+2)+j] = source_term(x[i], x[j]);
            //ref[i*(N+2)+j] = ref_func(x[i], x[j]);
        }
    }

    /*
    FILE *init_out;
    std::string init_fname("init_");
    init_fname.append(std::to_string(N));
    init_fname.append(".txt");
    init_out = fopen(init_fname.data(), "w");
    for(int i=0; i<N_grid; i++){
        //printf("%.18f ", d[i]);
        fprintf(init_out, "%.18f ", u[i]);
    }
    fclose(init_out);
    */


    double err, err_max;
    int iters=0;

    double Ad[N_grid] {0}, residual[N_grid] {0}, d[N_grid] {0};

    // initial residual = b-Ax
    # pragma omp parallel for collapse(2)
    for(int i=1; i<N+1; i++){
        for(int j=1; j<N+1; j++){
            residual[i*(N+2)+j] = source[i*(N+2)+j] -(u[(i+1)*(N+2)+j]
                                            +u[(i-1)*(N+2)+j]
                                            +u[i*(N+2)+j+1]
                                            +u[i*(N+2)+j-1]
                                            -4*u[i*(N+2)+j])/dx/dx;
            d[i*(N+2)+j] = residual[i*(N+2)+j];
        }
    }


    do{
        double alpha=0, alpha_num=0, alpha_den=0;
        double beta=0, beta_num=0;
        err=0;
        err_max=0;

        // compute Ad = A(d)
        # pragma omp parallel for collapse(2)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                Ad[i*(N+2)+j] = (d[(i+1)*(N+2)+j]
                                +d[(i-1)*(N+2)+j]
                                +d[i*(N+2)+j+1]
                                +d[i*(N+2)+j-1]
                                -4*d[i*(N+2)+j])/dx/dx;
            }
        }
        // alpha = r*r/d*Ad
        # pragma omp parallel for collapse(2) reduction(+:alpha_num, alpha_den)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                alpha_num += residual[i*(N+2)+j]*residual[i*(N+2)+j];
                alpha_den += d[i*(N+2)+j]*Ad[i*(N+2)+j];
            }
        }
        alpha = alpha_den!=0 ? alpha_num / alpha_den : 0;


        # pragma omp parallel for collapse(2) reduction(+:beta_num)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                u[i*(N+2)+j] += alpha*d[i*(N+2)+j];
                residual[i*(N+2)+j] -= alpha*Ad[i*(N+2)+j];
                beta_num += residual[i*(N+2)+j]*residual[i*(N+2)+j];
            }
        }
        beta = beta_num/alpha_num;

        # pragma omp parallel for collapse(2) reduction(+:err) reduction(max:err_max)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                d[i*(N+2)+j] = residual[i*(N+2)+j] + beta*d[i*(N+2)+j];
                err += fabs(residual[i*(N+2)+j]);
                //err += fabs(residual[i][j]/ref[i][j]);
                err_max = fabs(residual[i*(N+2)+j]) > err_max ? fabs(residual[i*(N+2)+j]) : err_max;
            }
        }

        err /= N*N;
        iters++;

    }while(err>terminate);
    printf("step %d, avg residual=%6e, max residual=%6e\n", iters, err, err_max);

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

    return 0;
}
