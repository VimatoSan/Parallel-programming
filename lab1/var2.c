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


int calculate_offlset(int rank, int size) {
	int offset = 0;
	int default_offset = N / size;
	for (int i = 0; i < rank; i++) {
		offset += calculate_chunk_size(i, size);
	}
	return offset;
}


int* create_offset_array(int size) {
	int* offset_array = (int*) malloc(sizeof(int) * size);
	for (int i = 0; i < size; ++i) {
		offset_array[i] = calculate_offlset(i, size);
	}
	return offset_array;
}


void vector_sub_vector(double* vector_a, double* vector_b, double *result, int size) {
	for (int i = 0; i < size; ++i) {
		result[i] = vector_a[i] - vector_b[i];
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

 
void init_chunk_vector_u(double* vector_u, int chunk_size, int offset) {
	for (int i = 0; i < chunk_size; ++i) {
		vector_u[i] = sin(2 * 3.1415 * (offset + i) / N);
	}
}


void init_null_vector(double *vector, int size) {
	for (int i = 0; i < size; ++i) {
		vector[i] = 0;
	}
}


void init_chunk_vector_x(double* vector_x, int size) {
	init_null_vector(vector_x, size);
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


void copy_vector(double* source, double* dest, int size) {
	for (int i = 0; i < size; ++i) {
		dest[i] = source[i];
	}
}


void matrix_mult_vector(double *matrix, double* vector, double* result, int* offset_array, int matrix_size, int rank, int size) {
	int offset;
	init_null_vector(result, matrix_size);
	double* recv_buffer = (double*) malloc(sizeof(double) * (N / size + 1));

	for (int proc_counter = 0; proc_counter < size; ++proc_counter) {
		
		offset = offset_array[(rank + proc_counter) % size];
	
		int vector_size = calculate_chunk_size((rank + proc_counter) % size, size);
		for (int i = 0; i < matrix_size; ++i) {
			for (int j = 0; j < vector_size; ++j)
			{
				result[i] += matrix[i * N + offset + j] * vector[j];
			}
		}

		// MPI_Sendrecv(vector, N / size + 1, MPI_DOUBLE, (rank + 1) % size, 123, recv_buffer, N / size + 1, MPI_DOUBLE, (rank + size - 1) % size, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// copy_vector(recv_buffer, vector, N / size + 1);
		//free(vector);
		//vector = recv_buffer;
		MPI_Sendrecv_replace(vector, N / size + 1, MPI_DOUBLE, (rank + size - 1) % size, 123, (rank + 1) % size, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	free(recv_buffer);
}


void collect_vector(double *chunk_vector, double* vector, int* offset_array, int chunk_size, int rank, int size) {

	int *recv_counts = (int *)malloc(sizeof(int) * size);
	for (int i = 0; i < size; i++) {
		recv_counts[i] = calculate_chunk_size(i, size);
	}

	MPI_Allgatherv(chunk_vector, chunk_size, MPI_DOUBLE, vector, recv_counts, offset_array, MPI_DOUBLE, MPI_COMM_WORLD);
	free(recv_counts);
}


void distribute_matrix(double* matrix, double* chunk_matrix, int* offset_array, int chunk_size, int rank, int size) {
	int *send_counts;
    int *offsets;

    if (rank == RANK_ROOT) {
        send_counts = (int *)malloc(sizeof(int) * size);
        offsets = (int *)malloc(sizeof(int) * size);

        for (int i = 0; i < size; i++) {
            send_counts[i] = calculate_chunk_size(i, size) * N;
			offsets[i] = offset_array[i] * N;
        }
    }
	MPI_Scatterv(matrix, send_counts, offsets, MPI_DOUBLE, chunk_matrix, chunk_size * N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
	if (rank == RANK_ROOT) {
		free(offsets);
		free(send_counts);
	}
}

double vector_difference(double* vector_a, double* vector_b, int size) {
	vector_sub_vector(vector_a, vector_b, vector_a, size);
	return calculate_vector_norm(vector_a, size);
}


void calculate_norm(double *vector, int size, double* result) {
	double sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += (vector[i] * vector[i]);
	}
	//printf("%lf\n", sum);
	MPI_Allreduce(&sum, result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	*result = sqrt(*result);
}


int main(int argc, char *argv[]) {
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double program_start_time = MPI_Wtime();

	
	int chunk_size = calculate_chunk_size(rank, size);
	int* offset_array = create_offset_array(size);

	double* matrix_A;
	double* chunk_vector_Ax = (double*)malloc(sizeof(double) * chunk_size);
	double* chunk_matrix_A = (double*)malloc(sizeof(double) * N * chunk_size);
	double* chunk_vector_b = (double*)malloc(sizeof(double) * chunk_size);
	double* chunk_vector_u = (double*)malloc(sizeof(double) * chunk_size);
	double* chunk_vector_x = (double*)malloc(sizeof(double) * N / size + 1);

	// init_chunk_vector_b(chunk_vector_b, chunk_size);
	init_chunk_vector_x(chunk_vector_x, chunk_size);
	init_chunk_vector_u(chunk_vector_u, chunk_size, offset_array[rank]); 
	if (rank == RANK_ROOT) {
		matrix_A = (double*)malloc(sizeof(double) * N * N);
		init_matrix_A(matrix_A);
	}
	distribute_matrix(matrix_A, chunk_matrix_A, offset_array, chunk_size, rank, size);
	if (rank == RANK_ROOT)
		free(matrix_A);
	
	matrix_mult_vector(chunk_matrix_A, chunk_vector_u, chunk_vector_b, offset_array, chunk_size, rank, size);
	free(chunk_vector_u);

	double numenator;
	double vector_b_norm;
	calculate_norm(chunk_vector_b, chunk_size, &vector_b_norm);
	double* temp_vector = (double*)malloc(sizeof(double) * chunk_size);

	double accuracy = BEGIN_ACCURACY;
	while (accuracy > EPSILON) {
		matrix_mult_vector(chunk_matrix_A, chunk_vector_x, chunk_vector_Ax, offset_array, chunk_size, rank, size);
		vector_sub_vector(chunk_vector_Ax, chunk_vector_b, temp_vector, chunk_size); // Ax - b

		calculate_norm(temp_vector, chunk_size, &numenator);

		accuracy = numenator / vector_b_norm;

		if (accuracy < EPSILON) {
			break;
		}
		if (rank == RANK_ROOT) {
			printf("%lf\n", accuracy);
		}
	
		vector_mult_scalar(temp_vector, TAU, temp_vector, chunk_size);
	
		vector_sub_vector(chunk_vector_x, temp_vector, chunk_vector_x, chunk_size); // Здесь создается кусок вектор x^(n+1)		
	}
	free(temp_vector);		

	double*	vector_x = (double*)malloc(sizeof(double) * N);
	double* vector_b = (double*)malloc(sizeof(double) * N);

	collect_vector(chunk_vector_x, vector_x, offset_array, chunk_size, rank, size);
	collect_vector(chunk_vector_b, vector_b, offset_array, chunk_size, rank, size);

	if (rank == RANK_ROOT) {
		print_matrix(vector_b, N, 1);
		print_matrix(vector_x, N, 1);
		printf("Result: %lf\nTime: %lf\n", vector_difference(vector_b, vector_x, N), MPI_Wtime() - program_start_time);
	}
	free(vector_b);
	free(vector_x);

	free(chunk_vector_b);
	free(chunk_matrix_A);
	free(chunk_vector_Ax);
	free(chunk_vector_x);
	MPI_Finalize(); 
	return 0;
}
