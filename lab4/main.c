#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>


#define RANK_ROOT 0
#define Dx 2
#define Dy 2
#define Dz 2
#define x0 -1
#define y0 -1
#define z0 -1
#define Nx 120
#define Ny 150
#define Nz 170
#define BEGIN_PHI 0
#define ALPHA 100000
#define EPSILON 0.00000001
#define BEGIN_ACCURACY 0.00000002

typedef struct {
    double x;
    double y;
    double z;
    int offset_z;
} steps;

int calculate_chunk_size(int rank, int size) {
	int chunk_size = Nz / size;
	if (rank < Nz % size) {
		chunk_size++;
	}
	return chunk_size;
}

void swap(double **a, double **b) {
    double *tmp = *a;
    *a = *b;
    *b = tmp;

}

void print_matrix(double* matrix, int width, int height) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j)
			printf("%lf ", matrix[i * width + j]);
		printf("\n");
	}
}

int calculate_offset(int rank, int size) {
	int offset = 0;
	for (int i = 0; i < rank; i++) {
		offset += calculate_chunk_size(i, size);
	}
	return offset;
}

double func_phi(double x, double y, double z) {
    return x*x + y*y + z*z;
}

double func_ro(double x, double y, double z) {
    return 6.0 - ALPHA * func_phi(x, y, z);
}

void fill_inner(double *space, int chunk_size) {
    for (int k = 1; k < chunk_size - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                space[k * Nx * Ny + j * Nx + i] = BEGIN_PHI; 
            }
        }
    }
}

void fill_borders(double *space, int chunk_size, steps *st) {
    for (int k = 0; k < chunk_size; ++k) { // i = 0
        for (int j = 0; j < Ny; ++j) {
            space[k * Nx * Ny + j * Nx] = func_phi(x0, y0 + j * st->y, z0 + (k + st->offset_z) * st->z); 
        }
    }
    for (int k = 0; k < chunk_size; ++k) { // i = Nx
        for (int j = 0; j < Ny; ++j) {
            space[k * Nx * Ny + j * Nx + Nx - 1] = func_phi(x0 + (Nx - 1) * st->x, y0 + j * st->y, z0 + (k + st->offset_z) * st->z); 
        }
    }
    for (int k = 0; k < chunk_size; ++k) { // j = 0
        for (int i = 0; i < Nx; ++i) {
            space[k * Nx * Ny + i] = func_phi(x0 + i * st->x, y0, z0 + (k + st->offset_z) * st->z); 
        }
    }
    for (int k = 0; k < chunk_size; ++k) { // j = Ny
        for (int i = 0; i < Nx; ++i) {
            space[k * Nx * Ny + (Ny - 1) * Nx + i] = func_phi(x0 + i * st->x, y0 + (Ny - 1) * st->y, z0 + (k + st->offset_z) * st->z); 
        }
    }
    for (int j = 0; j < Ny; ++j) { // k = 0
        for (int i = 0; i < Nx; ++i) {
            space[j * Nx + i] = func_phi(x0 + i * st->x, y0 + j * st->y, z0 + st->offset_z * st->z); 
        }
    }
    for (int j = 0; j < Ny; ++j) { // k = Nz
        for (int i = 0; i < Nx; ++i) {
            space[(chunk_size - 1) * Ny * Nx + j * Nx + i] = func_phi(x0 + i * st->x, y0 + j * st->y, z0 + (st->offset_z  + chunk_size - 1) * st->z); 
        }
    }
}

void init_buffers(double *lower_buffer, double *upper_buffer, int rank, int size) {
    if (rank == RANK_ROOT) {
        for (int i = 0; i < Nx * Ny; ++i)
            upper_buffer[i] = BEGIN_PHI;
    }
    else if (rank == size - 1) {
        for (int i = 0; i < Nx * Ny; ++i)
            lower_buffer[i] = BEGIN_PHI;
    }
    else {
        for (int i = 0; i < Nx * Ny; ++i) {
            upper_buffer[i] = BEGIN_PHI;
            lower_buffer[i] = BEGIN_PHI;
        }
    }
}

double my_abs(double value) {
    if (value < 0) {
        value = -value;
    }
    return value;
}

void calc_max_delta(double prev, double cur, double *max_delta) {
    if (my_abs(cur - prev) > *max_delta) {
        *max_delta = my_abs(cur - prev);
    }
}

double approximate_phi(int x, int y, int z, steps *st, double x_left, double x_right, double y_left, double y_right, double z_left, double z_right) {
    return (1.0 / ((2.0 / (st->x * st->x)) + (2.0 / (st->y * st->y)) + (2.0 / (st->z * st->z)) + ALPHA))
                * (((x_left + x_right) / (st->x * st->x)) + ((y_left + y_right) / (st->y * st->y)) + ((z_left + z_right) / (st->z * st->z))
                - func_ro(x0 + x * st->x, y0 + y * st->y, z0 + z * st->z));
}

void calc_lower_border(double *prev_space, double *cur_space, double *lower_buffer, MPI_Request *send_lower_buff, MPI_Request *recv_lower_buff, steps *st, double *max_delta, int rank) {
    double new_phi;
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            new_phi = approximate_phi(i, j, st->offset_z, st, prev_space[j * Nx + i - 1], prev_space[j * Nx + i + 1], 
                prev_space[(j - 1) * Nx + i], prev_space[(j + 1) * Nx + i], lower_buffer[j * Nx + i], prev_space[Nx * Ny + j * Nx + i]);

            cur_space[j * Nx + i] = new_phi;
            calc_max_delta(prev_space[j * Nx + i], new_phi, max_delta);
        }
    }
    MPI_Isend(&cur_space[0], Ny * Nx, MPI_DOUBLE, rank - 1, 123, MPI_COMM_WORLD, send_lower_buff);
    MPI_Irecv(lower_buffer, Nx * Ny, MPI_DOUBLE, rank - 1, 123, MPI_COMM_WORLD, recv_lower_buff);
}

void calc_upper_border(double *prev_space, double *cur_space, double *upper_buffer, MPI_Request *send_upper_buff, MPI_Request *recv_upper_buff, steps *st, double *max_delta, int chunk_size, int rank) {
    double new_phi;
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            new_phi = approximate_phi(i, j, st->offset_z + chunk_size - 1, st,
                prev_space[(chunk_size - 1) * Nx * Ny + j * Nx + i - 1], prev_space[(chunk_size - 1) * Nx * Ny + j * Nx + i + 1], 
                prev_space[(chunk_size - 1) * Nx * Ny + (j - 1)* Nx + i], prev_space[(chunk_size - 1) * Nx * Ny + (j + 1)* Nx + i],
                prev_space[(chunk_size - 2) * Nx * Ny + j * Nx + i], upper_buffer[j * Nx + i]);
            cur_space[(chunk_size - 1) * Ny * Nx + j * Nx + i] = new_phi;

            calc_max_delta(prev_space[(chunk_size - 1) * Ny * Nx + j * Nx + i], new_phi, max_delta);
        }
    }
    MPI_Isend(&cur_space[(chunk_size - 1) * Nx * Ny], Ny * Nx, MPI_DOUBLE, rank + 1, 123, MPI_COMM_WORLD, send_upper_buff);
    MPI_Irecv(upper_buffer, Nx * Ny, MPI_DOUBLE, rank + 1, 123, MPI_COMM_WORLD, recv_upper_buff);
}

void calc_inners(double *prev_space, double *cur_space, steps *st, int chunk_size, double *max_delta, int rank, int size) {
    double new_phi;
    for (int k = 1; k < chunk_size - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                new_phi = approximate_phi(i, j, st->offset_z + k, st,
                    prev_space[k * Nx * Ny + j * Nx + i - 1], prev_space[k * Nx * Ny + j * Nx + i + 1], 
                    prev_space[k * Nx * Ny + (j - 1) * Nx + i], prev_space[k * Nx * Ny + (j + 1)* Nx + i],
                    prev_space[(k - 1) * Nx * Ny + j * Nx + i], prev_space[(k + 1) * Nx * Ny + j * Nx + i]);
                cur_space[k * Nx * Ny + j * Nx + i] = new_phi; 
                calc_max_delta(prev_space[k * Ny * Nx + j * Nx + i], new_phi, max_delta);
            }
        }
    }
}

void wait_buffer_swap(MPI_Request *send_lower_buff, MPI_Request *send_upper_buff, MPI_Request *recv_lower_buff, MPI_Request *recv_upper_buff, int rank, int size) {
    if (rank != RANK_ROOT) {
        MPI_Wait(send_lower_buff, MPI_STATUS_IGNORE);
        MPI_Wait(recv_lower_buff, MPI_STATUS_IGNORE);
    }

    if (rank != size - 1) {
        MPI_Wait(send_upper_buff, MPI_STATUS_IGNORE);
        MPI_Wait(recv_upper_buff, MPI_STATUS_IGNORE);    
    }
}

void print_correct_phi(steps *st, int chunk_size) {
    for (int k = 0; k < chunk_size; k++) {
        printf("layer %d\n", k);
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; i++) {
                printf("%lf ", func_phi(x0 + i * st->x, y0 + j * st->y, z0 + (k + st->offset_z) * st->z));
            }
            printf("\n");
        }
    }
}

void print_space(double *subspace, int chunk_size) {
    for (int k = 0; k < chunk_size; ++k) {
        printf("layer - %d\n", k);
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                printf("%lf ", subspace[k * Nx * Ny + j * Nx + i]);
            }
            printf("\n");
        }
    }
}

void check_accuracy(double *prev_space, steps *st, int chunk_size, double *local_delta) {
    for (int k = 0; k < chunk_size; k++) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; i++) {
                calc_max_delta(prev_space[k * Nx * Ny + j * Nx + i], func_phi(x0 + i * st->x, y0 + j * st->y, z0 + (k + st->offset_z) * st->z), local_delta);
            }
        }
    }
}

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
    double *upper_buffer;
    double *lower_buffer;
    int chunk_size, size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double max_delta = BEGIN_ACCURACY;
    double local_delta = BEGIN_ACCURACY;
    steps st;
    st.x = (double)Dx / (Nx - 1); 
    st.y = (double)Dy / (Ny - 1); 
    st.z = (double)Dz / (Nz - 1); 
    st.offset_z = calculate_offset(rank, size);

    chunk_size = calculate_chunk_size(rank, size);
    if (rank == RANK_ROOT) {
        upper_buffer = malloc(sizeof(double) * Nx * Ny);
    }
    else if (rank == size - 1) {
        lower_buffer = malloc(sizeof(double) * Nx * Ny);
    }
    else {
        upper_buffer = malloc(sizeof(double) * Nx * Ny);
        lower_buffer = malloc(sizeof(double) * Nx * Ny);
    }
    double *prev_space = malloc(sizeof(double) * Nx * Ny * chunk_size);
    double *cur_space = malloc(sizeof(double) * Nx * Ny * chunk_size);
    fill_borders(prev_space, chunk_size, &st);
    fill_borders(cur_space, chunk_size, &st);
    fill_inner(prev_space, chunk_size);

    init_buffers(lower_buffer, upper_buffer, rank, size);

    MPI_Request send_upper_buff;
    MPI_Request send_lower_buff;
    MPI_Request recv_upper_buff;
    MPI_Request recv_lower_buff;

    while (max_delta > EPSILON) {
        local_delta = 0;
        if (rank != RANK_ROOT) 
            calc_lower_border(prev_space, cur_space, lower_buffer, &send_lower_buff, &recv_lower_buff, &st, &max_delta, rank);
        if (rank != size - 1)
            calc_upper_border(prev_space, cur_space, upper_buffer, &send_upper_buff, &recv_upper_buff, &st, &max_delta, chunk_size, rank);
        calc_inners(prev_space, cur_space, &st, chunk_size, &local_delta, rank, size);

        MPI_Allreduce(&local_delta, &max_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == RANK_ROOT)
            printf("accuracy - %.10lf\n", max_delta);
        swap(&cur_space, &prev_space); 
        wait_buffer_swap(&send_lower_buff, &send_upper_buff, &recv_lower_buff, &recv_upper_buff, rank, size);
    }

    local_delta = 0;
    check_accuracy(prev_space, &st, chunk_size, &local_delta);

    MPI_Allreduce(&local_delta, &max_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == RANK_ROOT)
        printf("Delta - %.10lf\n", max_delta);
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == 0) {
    //     print_space(prev_space, chunk_size);
    //     print_correct_phi(&st, chunk_size);
    // }

    free(prev_space);
    free(cur_space);
    if (rank == RANK_ROOT) {
        free(upper_buffer);
    }
    else if (rank == size - 1) {
        free(lower_buffer);
    }
    else {
        free(lower_buffer);
        free(upper_buffer);
    }
    MPI_Finalize(); 
}