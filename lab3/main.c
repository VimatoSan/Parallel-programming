#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define RANK_ROOT 0
#define N1 12
#define N2 7
#define N3 9

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%lf ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}


void init_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = (double)rand() / RAND_MAX;
        }
    }
}


int calculate_chunk_size(int data_size, int rank, int comm_size) {
	int chunk_size = data_size / comm_size;
	if (rank < data_size % comm_size) {
		chunk_size++;
	}
	return chunk_size;
}


void multiplicate_matrices(double *matrix_A, double* matrix_B, double* result, int rows_A, int cols_B, int size) {
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            double sum = 0;
            for (int k = 0; k < size; ++k) {
                sum += matrix_A[i * size + k] * matrix_B[j * size + k];
            }
            result[i * cols_B + j] = sum;
        }
    }
}

void decode_tag(int tag, int *rankx, int *ranky) {
    *rankx = 0;
    *ranky = 0;
    while (tag % 2 != 1) {
        tag /= 2;
        *rankx += 1;
    }
    while (tag % 3 != 1) {
        tag /= 3;
        *ranky += 1;
    } 
}


void collect_matrix(double *chunk_matrix, int matrix_rows, int matrix_cols, double* result_matrix, MPI_Comm comm, int rank, int rankx, int ranky, int comm_size, int sizex, int sizey) {
    MPI_Send(chunk_matrix, matrix_rows * matrix_cols, MPI_DOUBLE, RANK_ROOT, pow(2, rankx) * pow(3, ranky), comm);
    if (rank == RANK_ROOT) {
        int x, y;
        MPI_Status status;
        for (int i = 0; i < comm_size; ++i) {
            MPI_Probe(i, MPI_ANY_TAG, comm, &status);
            decode_tag(status.MPI_TAG, &x, &y);
            // Расчет адреса в буффере и создание вектора для приема данных
            int chunk_matrix_rows = calculate_chunk_size(N1, y, sizey);
            int chunk_matrix_cols = calculate_chunk_size(N3, x, sizex);

            int offset_y = 0, offset_x = 0;
            for (int j = 0; j < y; ++j) {
                offset_y += calculate_chunk_size(N1, j, sizey);
            }
            for (int j = 0; j < x; ++j) {
                offset_x += calculate_chunk_size(N3, j, sizex);
            }
            printf("%d, %d, %d, %d, %d, %d\n", x, y, chunk_matrix_rows, chunk_matrix_cols, offset_x, offset_y);

            // Создание вектора для приема данных
            MPI_Datatype submatrix, resized_submatrix;
            MPI_Type_vector(chunk_matrix_rows, chunk_matrix_cols, N3, MPI_DOUBLE, &submatrix);
            MPI_Type_commit(&submatrix);
            //MPI_Type_create_resized(submatrix, 0, chunk_matrix_cols * sizeof(double), &resized_submatrix);
            //MPI_Type_commit(&resized_submatrix);

            MPI_Recv(result_matrix + offset_y * N3 + offset_x, 1, submatrix, i, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&submatrix);
            // MPI_Type_free(&resized_submatrix);
        }
    }
    // int *recv_counts;
    // int *offsets;
    // if (rank == RANK_ROOT) {
    //     for (int i = 0; i < comm_size; ++i) {
    //         recv_counts[i] = 1;
            
    //     }
    // }
	// MPI_Gatherv(chunk_matrix, matrix_rows * matrix_cols, MPI_DOUBLE, result_matrix, recv_counts, offsets, type, RANK_ROOT, comm);
    // if (rank == RANK_ROOT) {
	// 	free(recv_counts);
	// 	free(offsets);
	// }
}


void distribute_matrix_rows(double* matrix, int matrix_rows, int matrix_cols, double* chunk_matrix, int chunk_size, MPI_Comm comm, int rank, int comm_size) {
	int *send_counts;
    int *offsets;
    if (rank == RANK_ROOT) {
        send_counts = (int *)malloc(sizeof(int) * comm_size);
        offsets = (int *)malloc(sizeof(int) * comm_size);
        offsets[0] = 0;
        for (int i = 0; i < comm_size; i++) {
            send_counts[i] = calculate_chunk_size(matrix_rows, i, comm_size) * matrix_cols;
            if (i > 0)
                offsets[i] = offsets[i - 1] + send_counts[i - 1];
        }
    }
	MPI_Scatterv(matrix, send_counts, offsets, MPI_DOUBLE, chunk_matrix, chunk_size * matrix_cols, MPI_DOUBLE, RANK_ROOT, comm);
	if (rank == RANK_ROOT) {
		free(send_counts);
		free(offsets);
	}
}


void distribute_matrix_cols(double* matrix, int matrix_rows, int matrix_cols, double* chunk_matrix, int chunk_size, MPI_Datatype type, MPI_Comm comm, int rank, int comm_size) {
    int *send_counts;
    int *offsets;
    // MPI_Aint lb, ext;
    // MPI_Type_get_extent(type, &lb, &ext);
    if (rank == RANK_ROOT) {
        send_counts = (int *)malloc(sizeof(int) * comm_size);   
        offsets = (int *)malloc(sizeof(int) * comm_size);
        offsets[0] = 0;
        for (int i = 0; i < comm_size; i++) {            
            send_counts[i] = calculate_chunk_size(matrix_cols, i, comm_size);
            if (i > 0)
                offsets[i] = offsets[i - 1] + send_counts[i - 1];
        }
        // for (int i = 0; i < comm_size; i++) {            
        //     send_counts[i] = 1;
        //     offsets[i] = i;
        // }
    }
	MPI_Scatterv(matrix, send_counts, offsets, type, chunk_matrix, chunk_size * matrix_rows, MPI_DOUBLE, RANK_ROOT, comm);
	if (rank == RANK_ROOT) {
		free(send_counts);
		free(offsets);
	}
}


int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
    int dims[2]={0,0}, periods[2]={0,0}, coords[2], reorder=1;
    int size, rank;
    // int prevy, prevx, nexty, nextx;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, 2, dims);
    int sizey = dims[0], sizex = dims[1];
    MPI_Comm comm2d, row, col; // коммуникаторы
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Comm_rank(comm2d, &rank);

    double *matrix_A;
    double *matrix_B;
    double *matrix_C;

    MPI_Cart_get(comm2d, 2, dims, periods, coords);
    int ranky = coords[0],  rankx = coords[1];
    int row_dims[2] = {0, 1}, col_dims[2] = {1, 0};

    MPI_Cart_sub(comm2d, col_dims, &col);
    MPI_Cart_sub(comm2d, row_dims, &row);

    int col_rank, row_rank;
    MPI_Comm_rank(col, &col_rank);
    MPI_Comm_rank(row, &row_rank);

    if (rank == RANK_ROOT) {
        matrix_A = (double*)malloc(sizeof(double) * N1 * N2);
        matrix_B = (double*)malloc(sizeof(double) * N2 * N3);
        matrix_C = (double*)malloc(sizeof(double) * N1 * N3);

        init_matrix(matrix_A, N1, N2);
        init_matrix(matrix_B, N2, N3);

        print_matrix(matrix_A, N1, N2);
        print_matrix(matrix_B, N2, N3);
    }
    int chunk_size_A = calculate_chunk_size(N1, ranky, sizey);
    int chunk_size_B = calculate_chunk_size(N3, rankx, sizex);

    double *chunk_matrix_A = (double*)malloc(sizeof(double) * chunk_size_A * N2);
    double *chunk_matrix_B = (double*)malloc(sizeof(double) * chunk_size_B * N2);
    double *chunk_matrix_C = (double*)malloc(sizeof(double) * chunk_size_A * chunk_size_B);

    // printf("%d, %d, %d\n", rank, col_rank, row_rank);

    if (rankx == RANK_ROOT) {
        distribute_matrix_rows(matrix_A, N1, N2, chunk_matrix_A, chunk_size_A, col, col_rank, sizey);
    }
    MPI_Bcast(chunk_matrix_A, chunk_size_A * N2, MPI_DOUBLE, RANK_ROOT, row);

    if (ranky == RANK_ROOT) {
        MPI_Datatype vector_B;
        MPI_Type_vector(N2, 1, N3, MPI_DOUBLE, &vector_B);
        MPI_Type_commit(&vector_B);
        MPI_Datatype resized_vector_B;
        MPI_Type_create_resized(vector_B, 0, sizeof(double), &resized_vector_B);
        MPI_Type_commit(&resized_vector_B);

        distribute_matrix_cols(matrix_B, N2, N3, chunk_matrix_B, chunk_size_B, resized_vector_B, row, rank, sizex);

        MPI_Type_free(&vector_B);
        MPI_Type_free(&resized_vector_B);
    }
    MPI_Bcast(chunk_matrix_B, chunk_size_B * N2, MPI_DOUBLE, RANK_ROOT, col);

    multiplicate_matrices(chunk_matrix_A, chunk_matrix_B, chunk_matrix_C, chunk_size_A, chunk_size_B, N2);

    // Сборка матрицы на главном узле

    // MPI_Datatype vector_C;
    // MPI_Type_vector(chunk_size_A, chunk_size_B, chunk_size_B, MPI_DOUBLE, &vector_C);
    // MPI_Datatype resized_vector_C;
    // MPI_Type_create_resized(vector_C, 0, chunk_size_B * sizeof(double), &resized_vector_C);
    // MPI_Type_commit(&resized_vector_C);
    collect_matrix(chunk_matrix_C, chunk_size_A, chunk_size_B, matrix_C, comm2d, rank, rankx, ranky, size, sizex, sizey);

    // MPI_Type_free(&vector_C);
    // MPI_Type_free(&resized_vector_C);   

    for (int i = 0; i < size; i++) { 
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank==i) {
            printf("Process: %d\n", rank);
            printf("matrix A\n");
            print_matrix(chunk_matrix_A, chunk_size_A, N2);
            printf("matrix B\n");
            print_matrix(chunk_matrix_B, chunk_size_B, N2);
            printf("matrix C\n");
            print_matrix(chunk_matrix_C, chunk_size_A, chunk_size_B);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == RANK_ROOT) {
        puts("Result");
        print_matrix(matrix_C, N1, N3);
    }
    
    free(chunk_matrix_A);
    free(chunk_matrix_B);
    free(chunk_matrix_C);

    if (rank == RANK_ROOT) {
        free(matrix_A);
        free(matrix_B);
        free(matrix_C);
    }
    MPI_Finalize(); 
}