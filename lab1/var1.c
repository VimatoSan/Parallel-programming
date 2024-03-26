#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define BEGIN_ACCURACY 1
#define N 1200
#define TAU 0.001
#define EPSILON 0.00001
#define RANK_ROOT 0


void print_matrix(double* matrix, int width, int height) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j)
			printf("%lf ", matrix[i * N + j]);
		printf("\n");
	}
}


int calculate_chunk_size(int rank, int size) {
	int chunk_size = N / size;
	if (rank < N % size) {
		chunk_size++;
	}
	return chunk_size;
}


int calculate_offset(int rank, int size) {
	int offset = 0;
	for (int i = 0; i < rank; i++) {
		offset += calculate_chunk_size(i, size);
	}
	return offset;
}


void vector_sub_vector(double* vector_a, double* vector_b, double *result, int offset_a, int offset_b, int size) {
	for (int i = 0; i < size; ++i) {
		result[i] = vector_a[offset_a + i] - vector_b[offset_b + i];
	}
}


void vector_mult_scalar(double* vector, double scalar, double* result, int size) {
	for (int i = 0; i < size; ++i) {
		result[i] = scalar * vector[i];
	}
}


double calculate_vector_norm(double *vector, int size) {
	double result = 0;
	for (int i = 0; i < size; ++i) {
		result += (vector[i] * vector[i]);
	}
	return sqrt(result);
}


void init_vector_u(double* vector_u) {
	for (int i = 0; i < N; ++i) {
		vector_u[i] = sin(2 * 3.1415 * i / N);
	}
}


void init_vector_x(double* vector_x) {
	for (int i = 0; i < N; ++i) {
		vector_x[i] = 0;
	}
}


void init_matrix_A(double* matrix_A) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j)  {
			if (i == j)
				matrix_A[i * N + j] = 2.0;
			else
				matrix_A[i * N + j] = 1.0;
		}
	}
}


void matrix_mult_vector(double *matrix, double* vector, double* result, int vector_size, int matrix_rows) {
	// print_matrix(matrix, N, matrix_rows);
	for (int i = 0; i < matrix_rows; ++i) {
		result[i] = 0;
		for (int j = 0; j < vector_size; ++j)
		{
			result[i] += matrix[i * vector_size + j] * vector[j];
		}	
	}
}


void distribute_matrix(double* matrix, double* chunk_matrix, int chunk_size, int rank, int size) {
	int *send_counts;
    int *offsets;
    if (rank == RANK_ROOT) {
        send_counts = (int *)malloc(sizeof(int) * size);
        offsets = (int *)malloc(sizeof(int) * size);
        offsets[0] = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = calculate_chunk_size(i, size) * N;
            if (i > 0)
                offsets[i] = offsets[i - 1] + send_counts[i - 1];
        }
    }
	MPI_Scatterv(matrix, send_counts, offsets, MPI_DOUBLE, chunk_matrix, chunk_size * N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
	if (rank == RANK_ROOT) {
		free(send_counts);
		free(offsets);
	}
}

// all processes send chunk_vector and chunks collect in every proc-ess
void collect_vector(double *chunk_vector, double* vector, int chunk_size, int size) {
	int *recv_counts = (int *)malloc(sizeof(int) * size);
	int *offsets = (int *)malloc(sizeof(int) * size);
	offsets[0] = 0;
	for (int i = 0; i < size; i++) {
		recv_counts[i] = calculate_chunk_size(i, size);
		if (i > 0)
			offsets[i] = offsets[i - 1] + recv_counts[i - 1];
	}

	MPI_Allgatherv(chunk_vector, chunk_size, MPI_DOUBLE, vector, recv_counts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
	free(recv_counts);
	free(offsets);
}

// chunk matrices multiplicate and result store on every process
void parallel_mult_matrix(double *matrix, double *vector, double *result_vector, int chunk_size, int size) {
	double* chunk_result_vector = (double*)malloc(sizeof(double) * chunk_size);
	matrix_mult_vector(matrix, vector, chunk_result_vector, N, chunk_size);
	collect_vector(chunk_result_vector, result_vector, chunk_size, size);
	free(chunk_result_vector);
}


void calculate_norm(double *vector, int size, double* result) {
	double sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += (vector[i] * vector[i]);
	}
	MPI_Allreduce(&sum, result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	*result = sqrt(*result);
}


double vector_difference(double* vector_a, double* vector_b, int size) {
	vector_sub_vector(vector_a, vector_b, vector_a, 0, 0, size);
	return calculate_vector_norm(vector_a, size);
}


int main(int argc, char *argv[]) {
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	double program_start_time = MPI_Wtime();


	int chunk_size = calculate_chunk_size(rank, size);
	int offset = calculate_offset(rank, size);
	double* matrix_A;
	double* chunk_vector_Ax = (double*)malloc(sizeof(double) * chunk_size);
	double* chunk_matrix_A = (double*)malloc(sizeof(double) * N * chunk_size);
	double* vector_b = (double*)malloc(sizeof(double) * N);;
	double* vector_u = (double*)malloc(sizeof(double) * N);
	double* vector_x = (double*)malloc(sizeof(double) * N);


	init_vector_x(vector_x);
	if (rank == RANK_ROOT) {
		matrix_A = (double*)malloc(sizeof(double) * N * N);
		init_matrix_A(matrix_A);
		init_vector_u(vector_u);
	}
	distribute_matrix(matrix_A, chunk_matrix_A, chunk_size, rank, size);
	if (rank == RANK_ROOT)
		free(matrix_A);

	MPI_Bcast(vector_u, N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

	parallel_mult_matrix(chunk_matrix_A, vector_u, vector_b, chunk_size, size); 	// Инициализируется вектор b
	free(vector_u);
	double vector_b_norm = calculate_vector_norm(vector_b, N);
	double numenator;

	double accuracy = BEGIN_ACCURACY;
	double* temp_vector = (double*)malloc(sizeof(double) * chunk_size);

	while (accuracy > EPSILON) {
		matrix_mult_vector(chunk_matrix_A, vector_x, chunk_vector_Ax, N, chunk_size);

		vector_sub_vector(chunk_vector_Ax, vector_b, temp_vector, 0, offset, chunk_size); // Ax - b

		calculate_norm(temp_vector, chunk_size, &numenator);

		accuracy = numenator / vector_b_norm;
		if (accuracy < EPSILON) {
			break;
		}
		if (rank == RANK_ROOT) {
			printf("%lf\n", accuracy);
		}
	
		vector_mult_scalar(temp_vector, TAU, temp_vector, chunk_size);
	
		vector_sub_vector(vector_x, temp_vector, temp_vector, offset, 0, chunk_size); // Здесь создается кусок вектор x^(n+1)
		
		collect_vector(temp_vector, vector_x, chunk_size, size);	// Сбор единого вектора x^(n+1)
	}
	free(temp_vector);		


	if (rank == RANK_ROOT) {
		print_matrix(vector_b, N, 1);
		print_matrix(vector_x, N, 1);
		printf("Result: %lf\nTime: %lf\n", vector_difference(vector_b, vector_x, N), MPI_Wtime() - program_start_time);
	}

	free(vector_b);
	free(chunk_matrix_A);
	free(chunk_vector_Ax);
	free(vector_x);
	MPI_Finalize(); 
	return 0;
}
