# Jacobi Method using CUDA
Calculate the Jacobi Method using CUDA for parallel programming course at UoA

## JacobiCuda:
To calculate Jacobi via CUDA with one GPU the program reads the input
and allocate the necessary memory in the GPU with the library function cudaMalloc()
for arrays uc, uc_old with size (n+2)\*(m+2). Similarly, memory is allocated for
variables of the Jacobi function. The number of threads is calculated by dividing the
number of datapoints by 500 and adding 1. With this implementation we make
appropriate number of blocks to execute Jacobi. So, a table is bound
f_error \[BLOCK_NUM \* 500 \], initializing the values via cudaMemcpy().Subsequently
one_jacobi_iteration() runs with BLOCK_NUM \* 500 threads, where one thread for each
series. In this function the program executes the Jacobi algorithm for each series,
assigning to y the value blockDim.x\*blockIdx.x+threadIdx.x, the location of the
thread in the specific block and saving the final error in its corresponding position
array f_error. Finally, the array f_error is copied to the CPU, where it is calculated
sum of its prices.

For two GPUs the implementation is similar to the first one, except that it is committed
memory for two uc, uc_old arrays practically splitting the center array in half as well
and for two arrays f_error\[BLOCK_NUM\*500\], where BLOCK_NUM is equal to (size/2)/500 for
the addition of new columns halo points. When executing the central while called
the createhalo function, which runs on a GPU to transfer data
to the new halo points. After this is done, for cudaSetDevice(0) it is called
one_jacobi_iteration () with BLOCK_NUM \* 500 threads and similarly with a GPU we copy the
table values in the CPU. Then we call cudaSetDevice(1) and
one_jacobi_iteration() synchronizing the two devices.


## Times:
![image](https://user-images.githubusercontent.com/75782840/148710245-2e26da9a-0816-4f45-822b-2a18668541cc.png)
![image](https://user-images.githubusercontent.com/75782840/148710257-07eaed3a-c221-4564-9fe9-969310a94b26.png)

From the above data we observe that the CUDA for larger numbers is ten of even thousand of times faster than all other techniques, as operations are performed within GPUs
is specialized in calculating operations such as Jacobi.
