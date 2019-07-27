/**
* Vincent Terpstra
* Sudoku.cu
* March 18 / 2019
* An Optimistic approach to solving a Sudoku on a CUDA enabled GPU
*    Assumes that the puzzle is deterministic(single solvable solution)
*        AND each next step can be found with the kernel
* KERNEL: educatedGuess
*   searches each square in a box for
*    squares that have only a single appropiate value
*    OR values that (in the box) can only fit in one square
*/

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
// CUDA header file
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>
#include <stdio.h> 
// UNASSIGNED is used for empty cells in sudoku grid 
#define UNASSIGNED 0 
// N is used for the size of Sudoku grid. Size will be NxN 
#define BOXWIDTH 3
#define N (BOXWIDTH * BOXWIDTH)

/*
 * kernel to solve a sudoku
 * Input: sudoku puzzle partitioned into boxes
 *	* d_a = the sudoku puzzle
 *	figures out what values can fit in each square
 *  figures out how many spots each value can go
 *  assigns the appropiate values,
 *	saves to addedIdx to show that there is a change
 */

__global__ void educatedGuess(int * d_a, int * addedIdx) {
	int idx = threadIdx.x + BOXWIDTH * threadIdx.y;
	int gridX = threadIdx.x + BOXWIDTH * blockIdx.x;
	int gridY = threadIdx.y + BOXWIDTH * blockIdx.y;
	int gridIdx = gridX + N * gridY;
	__shared__ bool hasValue[N]; //If the value occurs in the box
	__shared__ int  inBox[N];	 //Number of places each integer can go in the box
	hasValue[idx] = false;
	inBox[idx] = 0;
	__syncthreads();
	int at = d_a[gridIdx];
	if (at != 0)
		hasValue[at - 1] = true;
	__syncthreads();
	if (at != 0)
		return;
	//For remembering which values were seen in the rows and columns
	bool foundVal[N];
	for (int i = 0; i < N; ++i)
		foundVal[i] = hasValue[i];

	for (int check = 0; check < N; check++) {
		foundVal[d_a[N * check + gridX] - 1] = true;
		foundVal[d_a[N * gridY + check] - 1] = true;
	}
	int fndVals = 0;
	for (int i = 0; i < N; ++i)
		if (!foundVal[i]) {
			fndVals++;
			at = i + 1;
		}
	if (fndVals == 1) {
		//Only one possible value for this index
		d_a[gridIdx] = at;        //assign value
		addedIdx[0] = gridIdx;   //to tell host that the table has changed
		inBox[at - 1] = 4; //Prevent one index per value
	}
	__syncthreads();
	//Calculate the number of places each integer can go in the box
	for (int i = 0; i < N; ++i) {
		int num = (idx + i) % N; //keep each thread on a seperate idx
		if (!foundVal[num])
			inBox[num]++;
		__syncthreads();
	}
	for (int i = 0; i < N; ++i) {
		//if there is only one possible index for that value assign the value
		if (inBox[i] == 1 && !foundVal[i]) {
			d_a[gridIdx] = i + 1;    //assign value
			addedIdx[0] = gridIdx;   //to tell host that the table has changed
		}
	}
}

/* A utility function to print grid  */
void printGrid(int grid[N][N])
{
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++)
			printf("%3d", grid[row][col]);
		printf("\n");
	}
}
__global__ void superSolve(int * d_a) {
	__shared__ bool rowHas[N][N];
	__shared__ bool colHas[N][N];
	__shared__ bool boxHas[N][N];
	__shared__ int added, past;
	
	int row = threadIdx.x;
	int col = threadIdx.y;
	int box = row / BOXWIDTH + (col / BOXWIDTH) * BOXWIDTH;
	
	int gridIdx = col * N + row;
	int at = d_a[gridIdx];
	
	if (!gridIdx) { //only 0 needs to set changed
			added = -1;
			past  = -2;
		}
		rowHas[col][row] = false;
		colHas[col][row] = false;
		boxHas[col][row] = false;
	__syncthreads();

	if (at != UNASSIGNED) {
		rowHas[row][at - 1] = true;
		colHas[col][at - 1] = true;
		boxHas[box][at - 1] = true;
	}

	while (added != past) {
		__syncthreads();
		if(!gridIdx)
			past = added;
		if (at == 0) {
			int count = 0;
			for (int num = 0; num < N; ++num) {
				if (!(rowHas[row][num] || colHas[col][num] || boxHas[box][num])) {
					count++;
					at = num + 1;
				}
			}
			if (count == 1) {
				d_a[gridIdx] = at;
				rowHas[row][at - 1] = true;
				colHas[col][at - 1] = true;
				boxHas[box][at - 1] = true;
				added = gridIdx;
			} else {
				at = UNASSIGNED;
			}	
		}
		__syncthreads();
	}
}
/* Driver Program to test above functions */
int main()
{ /* 0 means unassigned cells */
   int grid[N][N] =
   { {3, 0, 6, 5, 0, 8, 4, 0, 0},
   {5, 2, 0, 0, 0, 0, 0, 0, 0},
   {0, 8, 7, 0, 0, 0, 0, 3, 1},
   {0, 0, 3, 0, 1, 0, 0, 8, 0},
   {9, 0, 0, 8, 6, 3, 0, 0, 5},
   {0, 5, 0, 0, 9, 0, 6, 0, 0},
   {1, 3, 0, 0, 0, 0, 2, 5, 0},
   {0, 0, 0, 0, 0, 0, 0, 7, 4},
   {0, 0, 5, 2, 0, 6, 3, 0, 0} };

   /**
   int grid[N][N] =
   {{0,  8,   0,  0,  0,  0,  0,  3,  0,  0,  0, 10,  9,  7, 11, 0},
   {0,  9,  15, 13,  0, 10,  0,  0,  2,  6,  8, 16,  0,  0,  0, 0},
   {0,  0,  16,  0, 15,  0,  8,  0,  9,  0,  0,  0,  6,  0,  2, 0},
   {1,  0,   2,  0,  9, 11,  4,  6, 15,  3,  5,  7,  0,  0, 12, 0},
   {16, 6,   4,  0,  5,  2,  0,  0,  1,  0,  0,  0, 11,  0,  0, 12},
   {5,  11,  0,  0,  0,  3,  0, 15,  0, 16,  0, 13,  0,  1,  0, 8},
   {0,  0,   3,  0,  0,  6, 11, 14,  0,  5,  7,  0,  0,  9,  0, 0},
   {0,  0,   0, 14,  8,  0, 10,  0,  0, 11, 12,  0,  0,  0,  0, 0},
   {0,  7,  13,  0,  0,  0,  0, 12,  0,  8,  9,  0,  0,  0,  3, 0},
   {0,  0,  11,  9,  0,  7,  0,  0,  0,  0,  0, 12,  0,  8, 16, 5},
   {0,  0,  10,  0, 11, 13,  0,  0,  0,  0,  0,  3, 12,  0,  6, 0},
   {0,  5,   0,  0, 10, 15,  0,  1,  7,  2,  0,  0, 14, 11,  0, 0},
   {0,  0,   5,  0,  0, 12, 14,  0,  0, 10,  0,  0, 15,  0,  0, 4},
   {9,  0,  14,  6,  0,  0,  1,  0, 16,  0,  2,  0,  3,  0, 13, 0},
   {8,  13,  0,  4,  0,  0,  0,  0, 12,  7,  3,  0,  0,  6,  0, 0},
   {0,  16, 12,  0,  0,  5,  0,  9,  0, 13, 14,  4,  1,  0,  0, 0} };
   /**

	int grid[N][N] =
	{ {1,  0,   4,  0, 25,  0, 19,  0,  0,  10,  21, 8,  0,  14, 0,  6,  12,   9,  0,  0,  0,  0,  0,  0,  5},{5,  0,  19, 23, 24,  0, 22,  12,  0,  0,  16, 6,  0,  20,  0,  18,  0,   25,  14,  13,  10, 11,  0,  1,  15},{0,  0,   0,  0,  0,  0,  21,  5,  0,  20,  11,  10,  0,  1,  0,  4,  8,   24,  23,  15,  18,  0,  16,  22,  19},

 {0, 7, 21, 8, 18, 0, 0, 0, 11, 0, 5, 0, 0, 24, 0, 0, 0, 17, 22, 1, 9, 6, 25, 0, 0}, {0, 13, 15, 0, 22, 14, 0, 18, 0, 16, 0, 0, 0, 4, 0, 0, 0, 19, 0, 0, 0, 24, 20, 21, 17}, {12, 0, 11, 0, 6, 0, 0, 0, 0, 15, 0, 0, 0, 0, 21, 25, 19, 0, 4, 0, 22, 14, 0, 20, 0}, {8, 0, 0, 21, 0, 16, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 17, 23, 18, 22, 0, 0, 0, 24, 6}, {4, 0, 14, 18, 7, 9, 0, 22, 21, 19, 0, 0, 0, 2, 0, 5, 0, 0, 0, 6, 16, 15, 0, 11, 12}, {22, 0, 24, 0, 23, 0, 0, 11, 0, 7, 0, 0, 4, 0, 14, 0, 2, 12, 0, 8, 5, 19, 0, 25, 9}, {20, 0, 0, 0, 5, 0, 0, 0, 0, 17, 9, 0, 12, 18, 0, 1, 0, 0, 7, 24, 0, 0, 0, 13, 4}, {13, 0, 0, 5, 0, 2, 23, 14, 4, 18, 22, 0, 17, 0, 0, 20, 0, 1, 9, 21, 12, 0, 0, 8, 11}, {14, 23, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 20, 25, 0, 3, 4, 13, 0, 11, 21, 9, 5, 18, 22}, {7, 0, 0, 11, 17, 20, 24, 0, 0, 0, 3, 4, 1, 12, 0, 0, 6, 14, 0, 5, 25, 13, 0, 0, 0}, {0, 0, 16, 9, 0, 17, 11, 7, 10, 25, 0, 0, 0, 13, 6, 0, 0, 18, 0, 0, 19, 4, 0, 0, 20}, {6, 15, 0, 19, 4, 13, 0, 0, 5, 0, 18, 11, 0, 0, 9, 8, 22, 16, 25, 10, 7, 0, 0, 0, 0}, {0, 0, 0, 2, 0, 0, 10, 19, 3, 0, 1, 0, 22, 9, 4, 11, 15, 0, 20, 0, 0, 8, 23, 0, 25}, {0, 24, 8, 13, 1, 0, 0, 4, 20, 0, 17, 14, 0, 0, 18, 0, 16, 22, 5, 0, 11, 0, 10, 0, 0}, {23, 10, 0, 0, 0, 0, 0, 0, 18, 0, 6, 0, 16, 0, 0, 17, 1, 0, 13, 0, 0, 3, 19, 12, 0}, {25, 5, 0, 14, 11, 0, 17, 0, 8, 24, 13, 0, 19, 23, 15, 9, 0, 0, 12, 0, 20, 0, 22, 0, 7}, {0, 0, 17, 4, 0, 22, 15, 0, 23, 11, 12, 25, 0, 0, 0, 0, 18, 8, 0, 7, 0, 0, 14, 0, 13}, {19, 6, 23, 22, 8, 0, 0, 1, 25, 4, 14, 2, 0, 3, 7, 13, 10, 11, 16, 0, 0, 0, 0, 0, 0}, {0, 4, 0, 17, 0, 3, 0, 24, 0, 8, 20, 23, 11, 10, 25, 22, 0, 0, 0, 12, 13, 2, 18, 6, 0}, {0, 0, 7, 16, 0, 0, 6, 17, 2, 21, 0, 18, 0, 0, 0, 19, 0, 0, 8, 0, 0, 0, 0, 4, 0}, {18, 9, 25, 1, 2, 11, 0, 0, 13, 22, 4, 0, 21, 0, 5, 0, 23, 7, 0, 0, 15, 0, 3, 0, 8}, {0, 21, 10, 0, 0, 12, 0, 20, 16, 0, 19, 0, 0, 0, 0, 15, 14, 4, 2, 18, 23, 25, 11, 7, 0} }; /**/
	/**/
	int* d_a;      //Table
	int* d_result; //Table change indicator
	cudaMalloc((void**)&d_a, N*N * sizeof(int));
	cudaMalloc((void**)&d_result, sizeof(int));

	//Copy Sudoku over
	cudaMemcpy(d_a, grid, N*N * sizeof(int), cudaMemcpyHostToDevice);

	//Solve the Sudoku
	dim3 block(N, N);
	superSolve << <1, block >> > (d_a);

	//Copy Sudoku back
	cudaMemcpy(grid, d_a, N*N * sizeof(int), cudaMemcpyDeviceToHost);
	printGrid(grid);

	cudaFree(d_a);
	cudaFree(d_result);

	return 0;

}