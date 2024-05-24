#include <cmath>
#include <cstdio>
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

    const int N=N_in;
    double L=1.0;
    double w=1.6905;
    double terminate=1e-12;


    double dx=L/N;
    double x[N+2] {0};
    double u[N+2][N+2] {0};
    double source[N+2][N+2] {0}, ref[N+2][N+2] {0};

    for(int i=1; i<N+2; i++){
        x[i] = (i-0.5)*dx;
    }

    for(int j=1; j<N+1; j++){
        u[N+1][j] = x[j]*(1-x[j]);
        //printf("%3e\n", u[N+1][j]);
    }

    for(int i=0; i<N+2; i++){
        for(int j=0; j<N+2; j++){
            source[i][j] = source_term(x[i], x[j]);
            ref[i][j] = ref_func(x[i], x[j]);
        }
    }

    double err, err_max;
    int iters=0;

    double Ad[N+2][N+2] {0}, residual[N+2][N+2] {0}, d[N+2][N+2] {0};

    // initial residual = b-Ax
    # pragma omp parallel for collapse(2)
    for(int i=1; i<N+1; i++){
        for(int j=1; j<N+1; j++){
            residual[i][j] = source[i][j] -(u[i+1][j]
                                            +u[i-1][j]
                                            +u[i][j+1]
                                            +u[i][j-1]
                                            -4*u[i][j])/dx/dx;
            d[i][j] = residual[i][j];
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
                Ad[i][j] = (d[i+1][j]
                            +d[i-1][j]
                            +d[i][j+1]
                            +d[i][j-1]
                            -4*d[i][j])/dx/dx;
            }
        }
        // alpha = r*r/d*Ad
        # pragma omp parallel for collapse(2) reduction(+:alpha_num, alpha_den)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                alpha_num += residual[i][j]*residual[i][j];
                alpha_den += d[i][j]*Ad[i][j];
            }
        }
        alpha = alpha_den!=0 ? alpha_num / alpha_den : 0;
        printf("alpha %f ", alpha);


        # pragma omp parallel for collapse(2) reduction(+:beta_num)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                residual[i][j] -= alpha*Ad[i][j];
                beta_num += residual[i][j]*residual[i][j];
            }
        }
        beta = beta_num/alpha_num;

        # pragma omp parallel for collapse(2) reduction(+:err) reduction(max:err_max)
        for(int i=1; i<N+1; i++){
            for(int j=1; j<N+1; j++){
                d[i][j] = residual[i][j] + beta*d[i][j];
                err += fabs(residual[i][j]);
                //err += fabs(residual[i][j]/ref[i][j]);
                err_max = fabs(residual[i][j]) > err_max ? fabs(residual[i][j]) : err_max;
            }
        }

        err /= N*N;
        iters++;
        printf("step %d, err=%6e, max err=%6e\n", iters, err, err_max);

    }while(err>terminate);

    return 0;
}
