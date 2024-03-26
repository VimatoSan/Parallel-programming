#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define BEGIN_ACCURACY 1
#define N 1100
#define TAU 0.001
#define EPSILON 0.00001


void print_matrix(double* matrix, int width, int height) {
	// # pragma omp parallel for
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j)
			printf("%lf ", matrix[i * N + j]);
		printf("\n");
	}
}


void vector_sub_vector(double* vector_a, double* vector_b, double *result) {
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		result[i] = vector_a[i] - vector_b[i];
	}
}


void vector_mult_scalar(double* vector, double scalar, double* result) {
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		result[i] = scalar * vector[i];
	}
}

double calculate_vector_norm(double *vector) {
	double result = 0;
	#pragma omp paralell for reduction (+:result)
	for (int i = 0; i < N; ++i) {
		result += (vector[i] * vector[i]);
	}
	return sqrt(result);
}


void init_vector_u(double* vector_u) {
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		vector_u[i] = sin(2 * 3.1415 * i / N);
	}
}


void init_vector_x(double* vector_x) {
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		vector_x[i] = 0;
	}
}


void init_matrix_A(double* matrix_A) {
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j)  {
			if (i == j)
				matrix_A[i * N + j] = 2.0;
			else
				matrix_A[i * N + j] = 1.0;
		}
	}
}


void matrix_mult_vector(double *matrix, double* vector, double* result) {
	// print_matrix(matrix, N, matrix_rows);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		result[i] = 0;
		for (int j = 0; j < N; ++j)
		{
			result[i] += matrix[i * N + j] * vector[j];
		}	
	}
}

double vector_difference(double* vector_a, double* vector_b) {
	vector_sub_vector(vector_a, vector_b, vector_a);
	return calculate_vector_norm(vector_a);
}


int main(int argc, char *argv[]) {
	struct timespec start, end;
	
	double* matrix_A = (double*)malloc(sizeof(double) * N * N);
	double* vector_b = (double*)malloc(sizeof(double) * N);
	double* vector_u = (double*)malloc(sizeof(double) * N);
	double* vector_x = (double*)malloc(sizeof(double) * N);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	init_matrix_A(matrix_A);
	init_vector_x(vector_x);
	init_vector_u(vector_u);
	matrix_mult_vector(matrix_A, vector_u, vector_b);
	free(vector_u);
	double vector_b_norm = calculate_vector_norm(vector_b);

	double numenator;

	double accuracy = BEGIN_ACCURACY;
	double* temp_vector = (double*)malloc(sizeof(double) * N);
	while (accuracy > EPSILON) {
		matrix_mult_vector(matrix_A, vector_x, temp_vector);
		vector_sub_vector(temp_vector, vector_b, temp_vector); // Ax - b

		numenator = calculate_vector_norm(temp_vector);
		accuracy = numenator / vector_b_norm;
		if (accuracy < EPSILON) {
			break;
		}
		printf("%lf\n", accuracy);
		vector_mult_scalar(temp_vector, TAU, temp_vector);
	
		vector_sub_vector(vector_x, temp_vector, vector_x); 
	}
	free(temp_vector);		
	
	print_matrix(vector_b, N, 1);
	print_matrix(vector_x, N, 1);
	printf("Result: %lf\n", vector_difference(vector_b, vector_x));


	free(matrix_A);
	free(vector_b);
	free(vector_x);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Time taken: %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
	return 0;
}
