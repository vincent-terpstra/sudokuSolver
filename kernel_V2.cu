#include <stdio.h>
// CUDA header file
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
// UNASSIGNED is used for empty cells in Sudoku grid 
#define UNASSIGNED 0
// BOX_W is used for the length of one of the square sub-regions of the Sudoku grid.
// Overall length will be N * N.
#define BOX_W 5
#define N (BOX_W * BOX_W)


__global__ void solve(int* d_a) {
	// Used to remember which row | col | box ( section ) have which values
	__shared__ bool rowHas[N][N];
	__shared__ bool colHas[N][N];
	__shared__ bool boxHas[N][N];
	// Used to ensure that the table has changed
	__shared__ bool changed;
	
	// Number of spaces which can place the number in each section
	__shared__ char rowCount[N][N];
	__shared__ char colCount[N][N];
	__shared__ char boxCount[N][N];
	// Where the square is located in the Sudoku
	char row = threadIdx.x;
	char col = threadIdx.y;
	int box = row / BOX_W + (col / BOX_W) * BOX_W;

	int gridIdx = col * N + row;
	char at = d_a[gridIdx];
	
	// Unique identifier for each square in row, col, box
	// Corresponds to the generic Sudoku Solve
	// Using a Sudoku to solve a Sudoku !!!
	int offset = col + (row % BOX_W) * BOX_W + (box % BOX_W);
	int m_offset = offset + N;
	// Square's location in the Sudoku

	char count = 0; //Number of values which can fit in this square
	int notSeen = 0; //Boolean Array as an Integer

	if (gridIdx == 0) changed = true;
	rowHas[col][row] = false;
	colHas[col][row] = false;
	boxHas[col][row] = false;

	rowCount[col][row] = 0;
	colCount[col][row] = 0;
	boxCount[col][row] = 0;

	__syncthreads();
	if (at != UNASSIGNED) {
		rowHas[row][at - 1] = true;
		colHas[col][at - 1] = true;
		boxHas[box][at - 1] = true;
	}
	__syncthreads();
	
	int b_shuttle = 1;
	for (int idx = offset; idx < m_offset; ++idx) {
		int num = idx % N;
		if (at == UNASSIGNED && !(rowHas[row][num] || boxHas[box][num] || colHas[col][num])) {
			notSeen |= b_shuttle;	//this value can go here
			++count;				//how many values this square can have

			//how many values this section can have
			rowCount[row][num]++;
			colCount[col][num]++;
			boxCount[box][num]++;
		}
		__syncthreads();
		b_shuttle <<= 1;
	}
	if (at == UNASSIGNED && count == 0) //NOT POSSIBLE SUDOKU
		changed = false;
	__syncthreads();
	
	
	// Previous loop has not changed any values
	
	while(changed){
		__syncthreads();
		bool inSection = true;
		if (gridIdx == 0) // forget previous change
			changed = false;
		int guess = 0; // last value found which can fit in this square
		
		int b_shuttle = 1;
		for (int idx = offset; idx < m_offset; ++idx) {
			// Ensures that every square in each section is working on a different number in the section
			int num = idx % N;
			if (b_shuttle & notSeen) {
				if (rowHas[row][num] || boxHas[box][num] || colHas[col][num]) {
					notSeen ^= b_shuttle;
					--count;
					rowCount[row][num]--;
					colCount[col][num]--;
					boxCount[box][num]--;
				} else if (inSection) {
					guess = num;
				}
			}
			__syncthreads();
			if ((b_shuttle & notSeen) && (rowCount[row][num] == 1 || boxCount[box][num] == 1 || colCount[col][num] == 1))
				inSection = false;

			b_shuttle <<= 1;
		}

		if (count == 1 || !inSection) {
			at = guess + 1;
			notSeen = count = 0;
			rowHas[row][guess] = true;
			colHas[col][guess] = true;
			boxHas[box][guess] = true;
			changed = true;
		}
		__syncthreads();
	};

	if (!(rowHas[row][col] && colHas[row][col] && boxHas[box][col]))
		changed = true; //HAVE NOT SOLVED the sudoku
	if (changed && gridIdx == 0)
		at = 0;
	d_a[gridIdx] = at;
}

void print(int result[N][N]) {
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++)
			printf("%3d", result[row][col]);
		printf("\n");
	}
}

// Driver program to test main program functions
int main() {
	int h_a[N][N] = {
	  {  1,  0,  4,  0, 25,  0, 19,  0,  0, 10, 21,  8,  0, 14,  0,  6, 12,  9,  0,  0,  0,  0,  0,  0,  5},
	  {  5,  0, 19, 23, 24,  0, 22, 12,  0,  0, 16,  6,  0, 20,  0, 18,  0, 25, 14, 13, 10, 11,  0,  1, 15},
	  {  0,  0,  0,  0,  0,  0, 21,  5,  0, 20, 11, 10,  0,  1,  0,  4,  8, 24, 23, 15, 18,  0, 16, 22, 19},
	  {  0,  7, 21,  8, 18,  0,  0,  0, 11,  0,  5,  0,  0, 24,  0,  0,  0, 17, 22,  1,  9,  6, 25,  0,  0},
	  {  0, 13, 15,  0, 22, 14,  0, 18,  0, 16,  0,  0,  0,  4,  0,  0,  0, 19,  0,  0,  0, 24, 20, 21, 17},
	  { 12,  0, 11,  0,  6,  0,  0,  0,  0, 15,  0,  0,  0,  0, 21, 25, 19,  0,  4,  0, 22, 14,  0, 20,  0},
	  {  8,  0,  0, 21,  0, 16,  0,  0,  0,  2,  0,  3,  0,  0,  0,  0, 17, 23, 18, 22,  0,  0,  0, 24,  6},
	  {  4,  0, 14, 18,  7,  9,  0, 22, 21, 19,  0,  0,  0,  2,  0,  5,  0,  0,  0,  6, 16, 15,  0, 11, 12},
	  { 22,  0, 24,  0, 23,  0,  0, 11,  0,  7,  0,  0,  4,  0, 14,  0,  2, 12,  0,  8,  5, 19,  0, 25,  9},
	  { 20,  0,  0,  0,  5,  0,  0,  0,  0, 17,  9,  0, 12, 18,  0,  1,  0,  0,  7, 24,  0,  0,  0, 13,  4},
	  { 13,  0,  0,  5,  0,  2, 23, 14,  4, 18, 22,  0, 17,  0,  0, 20,  0,  1,  9, 21, 12,  0,  0,  8, 11},
	  { 14, 23,  0, 24,  0,  0,  0,  0,  0,  0,  0,  0, 20, 25,  0,  3,  4, 13,  0, 11, 21,  9,  5, 18, 22},
	  {  7,  0,  0, 11, 17, 20, 24,  0,  0,  0,  3,  4,  1, 12,  0,  0,  6, 14,  0,  5, 25, 13,  0,  0,  0},
	  {  0,  0, 16,  9,  0, 17, 11,  7, 10, 25,  0,  0,  0, 13,  6,  0,  0, 18,  0,  0, 19,  4,  0,  0, 20},
	  {  6, 15,  0, 19,  4, 13,  0,  0,  5,  0, 18, 11,  0,  0,  9,  8, 22, 16, 25, 10,  7,  0,  0,  0,  0},
	  {  0,  0,  0,  2,  0,  0, 10, 19,  3,  0,  1,  0, 22,  9,  4, 11, 15,  0, 20,  0,  0,  8, 23,  0, 25},
	  {  0, 24,  8, 13,  1,  0,  0,  4, 20,  0, 17, 14,  0,  0, 18,  0, 16, 22,  5,  0, 11,  0, 10,  0,  0},
	  { 23, 10,  0,  0,  0,  0,  0,  0, 18,  0,  6,  0, 16,  0,  0, 17,  1,  0, 13,  0,  0,  3, 19, 12,  0},
	  { 25,  5,  0, 14, 11,  0, 17,  0,  8, 24, 13,  0, 19, 23, 15,  9,  0,  0, 12,  0, 20,  0, 22,  0,  7},
	  {  0,  0, 17,  4,  0, 22, 15,  0, 23, 11, 12, 25,  0,  0,  0,  0, 18,  8,  0,  7,  0,  0, 14,  0, 13},
	  { 19,  6, 23, 22,  8,  0,  0,  1, 25,  4, 14,  2,  0,  3,  7, 13, 10, 11, 16,  0,  0,  0,  0,  0,  0},
	  {  0,  4,  0, 17,  0,  3,  0, 24,  0,  8, 20, 23, 11, 10, 25, 22,  0,  0,  0, 12, 13,  2, 18,  6,  0},
	  {  0,  0,  7, 16,  0,  0,  6, 17,  2, 21,  0, 18,  0,  0,  0, 19,  0,  0,  8,  0,  0,  0,  0,  4,  0},
	  { 18,  9, 25,  1,  2, 11,  0,  0, 13, 22,  4,  0, 21,  0,  5,  0, 23,  7,  0,  0, 15,  0,  3,  0,  8},
	  {  0, 21, 10,  0,  0, 12,  0, 20, 16,  0, 19,  0,  0,  0,  0, 15, 14,  4,  2, 18, 23, 25, 11,  7,  0}
	};
	int* d_a;      //Table
	cudaMalloc((void**)&d_a, N * N * sizeof(int));
	// Copy Sudoku to device
	cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
	dim3 dBlock(N, N);
	solve << <1, dBlock >> > (d_a);
	// Copy Sudoku back to host
	cudaMemcpy(h_a, d_a, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	// Check if solved
	if (*h_a)
		print(h_a);
	else
		printf("No solution could be found.");
	cudaFree(d_a);
	return 0;
}