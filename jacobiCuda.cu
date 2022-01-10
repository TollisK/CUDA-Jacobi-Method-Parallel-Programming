/************************************************************
 * Program to solve a finite difference
 * discretization of the screened Poisson equation:
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * with zero Dirichlet boundary condition using the iterative
 * Jacobi method with overrelaxation.
 *
 * RHS (source) function
 *   f(x,y) = -alpha*(1-x^2)(1-y^2)-2*[(1-x^2)+(1-y^2)]
 *
 * Analytical solution to the PDE
 *   u(x,y) = (1-x^2)(1-y^2)
 *
 * Current Version: Christian Iwainsky, RWTH Aachen University
 * MPI C Version: Christian Terboven, RWTH Aachen University, 2006
 * MPI Fortran Version: Dieter an Mey, RWTH Aachen University, 1999 - 2005
 * Modified: Sanjiv Shah,        Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux,  Kuck and Associates, Inc. (KAI), 1998
 *
 * Unless READ_INPUT is defined, a meaningful input dataset is used (CT).
 *
 * Input : n     - grid dimension in x direction
 *         m     - grid dimension in y direction
 *         alpha - constant (always greater than 0.0)
 *         tol   - error tolerance for the iterative solver
 *         relax - Successice Overrelaxation parameter
 *         mits  - maximum iterations for the iterative solver
 *
 * On output
 *       : u(n,m)       - Dependent variable (solution)
 *       : f(n,m,alpha) - Right hand side function
 *
 *************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

/*************************************************************
 * Performs one iteration of the Jacobi method and computes
 * the residual value.
 *
 * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
 * are BOUNDARIES and therefore not part of the solution.
 *************************************************************/
__global__ void createhalo(double *array1,double *array2,int *maxXCount){
#define AR1(XX,YY) array1[(YY)*(*maxXCount)+(XX)]
#define AR2(XX,YY) array2[(YY)*(*maxXCount)+(XX)]
    for(int i = 0 ; i < (*maxXCount); i ++){
        AR1(i,(*maxXCount)/2 + 1) = AR2(i,1);
        AR2(i,0) = AR1(i,(*maxXCount)/2);
    }
}
__global__ void one_jacobi_iteration(double *xStart, double *yStart,
                            int *maxXCount, int *maxYCount,
                            double *src, double *dst,
                            double *deltaX, double *deltaY,
                            double *alpha, double *omega,double *f_error)
{
    if(blockDim.x*blockIdx.x+threadIdx.x > (*maxYCount)){
      f_error[blockDim.x*blockIdx.x+threadIdx.x] = 0.0;
    }
    else{
        // printf("If = %d\n",blockDim.x*blockIdx.x+threadIdx.x);
        double error = 0.0;
        __syncthreads();

    #define SRC(XX,YY) src[(YY)*(*maxXCount)+(XX)]
    #define DST(XX,YY) dst[(YY)*(*maxXCount)+(XX)]
        int x, y;
        double fX, fY;
        double updateVal;
        double f;

        // Coefficients
        double cx = 1.0/((*deltaX)*(*deltaX));
        double cy = 1.0/((*deltaY)*(*deltaY));
        double cc = -2.0*cx-2.0*cy-(*alpha);
        y = blockDim.x*blockIdx.x+threadIdx.x+1;
        fY = (*yStart) + (y-1)*(*deltaY);
        for (x = 1; x < ((*maxXCount)-1); x++)
        {
            fX = (*xStart) + (x-1)*(*deltaX);
            f = -(*alpha)*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY);
            updateVal = ((SRC(x-1,y) + SRC(x+1,y))*cx +
                            (SRC(x,y-1) + SRC(x,y+1))*cy +
                            SRC(x,y)*cc - f
                        )/cc;
            DST(x,y) = SRC(x,y) - (*omega)*updateVal;
            error += updateVal*updateVal;
        }
        // printf("errir %d %g\n",blockDim.x*blockIdx.x+threadIdx.x,error);
        f_error[blockDim.x*blockIdx.x+threadIdx.x] = error;
    }
}


/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}


int main(int argc, char **argv)
{
    int flag = 1;
    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double error;
    double *u_old;//, *tmp;
    int allocCount;
    int iterationCount, maxIterationCount;
//    printf("Input n,m - grid dimension in x,y direction:\n");
    scanf("%d,%d", &n, &m);
//    printf("Input alpha - Helmholtz constant:\n");
    scanf("%lf", &alpha);
//    printf("Input relax - successive over-relaxation parameter:\n");
    scanf("%lf", &relax);
//    printf("Input tol - error tolerance for the iterrative solver:\n");
    scanf("%lf", &tol);
//    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", &mits);

    int size_n = n+2;
    int size_m = m+2;

    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);

    allocCount = (n+2)*(m+2);

    u_old = (double*)calloc(allocCount, sizeof(double));

    double *tmpc;

    double *uc1;
    double *uc2;
    double *uc_old1;
    double *uc_old2;
    double *uc;
    double *uc_old;
    if(flag){
        cudaMalloc((void**)&uc1, (n+2)*(m/2+2)*sizeof(double));
        cudaMalloc((void**)&uc2, (n+2)*(m/2+2)*sizeof(double));

        cudaMalloc((void**)&uc_old1, (n+2)*(m/2+2)*sizeof(double));
        cudaMalloc((void**)&uc_old2, (n+2)*(m/2+2)*sizeof(double));
    }
    else{
        
        cudaMalloc((void**)&uc, allocCount*sizeof(double));

        cudaMalloc((void**)&uc_old, allocCount*sizeof(double));
    }
    double *xLeftc;
    double *yBottomc;
    int *size_nc;
    int *size_mc;
    double *deltaXc;
    double *deltaYc;
    double *alphac;
    double *relaxc;

    cudaMalloc((void**)&xLeftc,sizeof(double));
    cudaMalloc((void**)&yBottomc,sizeof(double));
    cudaMalloc((void**)&size_nc,sizeof(int));
    cudaMalloc((void**)&size_mc,sizeof(int));
    cudaMalloc((void**)&deltaXc,sizeof(double));
    cudaMalloc((void**)&deltaYc,sizeof(double));
    cudaMalloc((void**)&alphac,sizeof(double));
    cudaMalloc((void**)&relaxc,sizeof(double));


    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    iterationCount = 0;
    error = HUGE_VAL;
    
    cudaMemcpy(xLeftc,&xLeft,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(yBottomc,&yBottom,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(size_nc,&size_n,sizeof(int),cudaMemcpyHostToDevice);
    if(flag){
        size_m = m/2+2;
        cudaMemcpy(size_mc,&size_m,sizeof(int),cudaMemcpyHostToDevice);
    }
    else
        cudaMemcpy(size_mc,&size_m,sizeof(int),cudaMemcpyHostToDevice);
        
    cudaMemcpy(deltaXc,&deltaX,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(deltaYc,&deltaY,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(alphac,&alpha,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(relaxc,&relax,sizeof(double),cudaMemcpyHostToDevice);

    /* Iterate as long as it takes to meet the convergence criterion */

    int blocknum;
    double final_error;

    double *f_error1;
    double *f_error2;
    double *f_error;
    if(flag){
        blocknum = (m/2)/500;
        blocknum++;
        cudaMalloc((void**)&f_error1,(blocknum*500)*sizeof(double));
        cudaMalloc((void**)&f_error2,(blocknum*500)*sizeof(double));
    }
    else{
        blocknum = m/500;
        blocknum++;
        cudaMalloc((void**)&f_error,(blocknum*500)*sizeof(double));
    }

    double error2[blocknum*500];
    double error3[(((m/2)/500)+1)*500];

    clock_t start = clock(), diff;
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    { 
        final_error = 0.0;
        if(flag){
            createhalo<<<1,1>>>(uc_old1,uc_old2,size_mc);
            cudaSetDevice(0);
            one_jacobi_iteration<<<blocknum,500>>>(xLeftc, yBottomc,
                                     size_nc, size_mc,
                                     uc_old1, uc1,
                                     deltaXc, deltaYc,
                                     alphac, relaxc,f_error1);
            cudaMemcpy(error3,f_error1,(blocknum*500)*sizeof(double),cudaMemcpyDeviceToHost);
            for(int i = 0; i < (m/2);i++){
                final_error+=error3[i];
            }
            cudaSetDevice(1);
            one_jacobi_iteration<<<blocknum,500>>>(xLeftc, yBottomc,
                                     size_nc, size_mc,
                                     uc_old2, uc2,
                                     deltaXc, deltaYc,
                                     alphac, relaxc,f_error2);                      
            cudaDeviceSynchronize();

            cudaMemcpy(error3,f_error2,(blocknum*500)*sizeof(double),cudaMemcpyDeviceToHost);
            for(int i = 0 ;i < (m/2);i++){
                final_error+=error3[i];
            }
            tmpc = uc_old1;
            uc_old1 = uc1;
            uc1 = tmpc;

            tmpc = uc_old2;
            uc_old2 = uc2;
            uc2 = tmpc;
        }
        else{
            one_jacobi_iteration<<<blocknum,500>>>(xLeftc, yBottomc,
                                        size_nc, size_mc,
                                        uc_old, uc,
                                        deltaXc, deltaYc,
                                        alphac, relaxc,f_error);
            cudaMemcpy(error2,f_error,(blocknum*500)*sizeof(double),cudaMemcpyDeviceToHost);
            int i;
            for(i = 0 ;i < (n);i++){
                final_error+=error2[i];
            }
            tmpc = uc_old;
            uc_old = uc;
            uc = tmpc;
        }
        error = sqrt(final_error)/((n)*(m));
        iterationCount++;
        // Swap the buffers
    }
    
    
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Residual %g\n",error);

    // u_old holds the solution after the most recent buffers swap
    cudaMemcpy(u_old,uc_old,allocCount*sizeof(double),cudaMemcpyDeviceToHost);
    double absoluteError = checkSolution(xLeft, yBottom,
                                         n+2, m+2,
                                         u_old,
                                         deltaX, deltaY,
                                         alpha);
    printf("The error of the iterative solution is %g\n", absoluteError);

    cudaFree(uc);
    cudaFree(uc_old);
    return 0;
}
