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
#define Nx 150
#define Ny 100
#define Nz 120    
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

void fill_inner(double *subspace, int chunk_size) {
    for (int k = 1; k < chunk_size - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                subspace[k * Nx * Ny + j * Nx + i] = BEGIN_PHI; 
            }
        }
    }
}

void fill_borders(double *subspace, int chunk_size, steps *st) {
    for (int k = 0; k < chunk_size; ++k) { // i = 0
        for (int j = 0; j < Ny; ++j) {
            subspace[k * Nx * Ny + j * Nx] = func_phi(x0, y0 + j * st->y, z0 + (k + st->offset_z) * st->z); 
        }
    }
    for (int k = 0; k < chunk_size; ++k) { // i = Nx
        for (int j = 0; j < Ny; ++j) {
            subspace[k * Nx * Ny + j * Nx + Nx - 1] = func_phi(x0 + (Nx - 1) * st->x, y0 + j * st->y, z0 + (k + st->offset_z) * st->z); 
        }
    }
    for (int k = 0; k < chunk_size; ++k) { // j = 0
        for (int i = 0; i < Nx; ++i) {
            subspace[k * Nx * Ny + i] = func_phi(x0 + i * st->x, y0, z0 + (k + st->offset_z) * st->z); 
        }
    }
    for (int k = 0; k < chunk_size; ++k) { // j = Ny
        for (int i = 0; i < Nx; ++i) {
            subspace[k * Nx * Ny + (Ny - 1) * Nx + i] = func_phi(x0 + i * st->x, y0 + (Ny - 1) * st->y, z0 + (k + st->offset_z) * st->z); 
        }
    }
    for (int j = 0; j < Ny; ++j) { // k = 0
        for (int i = 0; i < Nx; ++i) {
            subspace[j * Nx + i] = func_phi(x0 + i * st->x, y0 + j * st->y, z0 + st->offset_z * st->z); 
        }
    }
    for (int j = 0; j < Ny; ++j) { // k = Nz
        for (int i = 0; i < Nx; ++i) {
            subspace[(chunk_size - 1) * Ny * Nx + j * Nx + i] = func_phi(x0 + i * st->x, y0 + j * st->y, z0 + (st->offset_z  + chunk_size) * st->z); 
        }
    }
}

void init_buffers(double *lower_buffer, double *upper_buffer, int rank, int size) {
    if (rank == RANK_ROOT) {
        for (int i = 0; i < (Nx - 2) * (Ny - 2); ++i)
            upper_buffer[i] = BEGIN_PHI;
    }
    else if (rank == size - 1) {
        for (int i = 0; i < (Nx - 2) * (Ny - 2); ++i)
            lower_buffer[i] = BEGIN_PHI;
    }
    else {
        for (int i = 0; i < (Nx - 2) * (Ny - 2); ++i) {
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

// double approximate_phi(double *subspace, int i, int j, int k,steps *st ) {
//     return (1 / ((2.0 / (st->x * st->x)) + (2.0 / (st->y * st->y)) + (2.0 / (st->z * st->z)) + ALPHA))
//                 * (((subspace[k * Nx * Ny + j * Nx + i + 1] + subspace[k * Nx * Ny + j * Nx + i - 1]) / (st->x * st->x)) 
//                 + ((subspace[k * Nx * Ny + (j + 1)* Nx + i] + subspace[k * Nx * Ny + (j - 1) * Nx + i]) / (st->y * st->y)) 
//                 + ((subspace[(k + 1) * Nx * Ny + j * Nx + i] + subspace[(k - 1) * Nx * Ny + j * Nx + i]) / (st->z * st->z)) - func_ro(x0 + i * st->x, y0 + j * st->y, z0 + (k + st->offset_z) * st->z));  
// }


void calc_borders(double *subspace, double *buffer_space, double *lower_buffer, double *upper_buffer, MPI_Request *send_lower_buff, MPI_Request *send_upper_buff, MPI_Datatype *inner_layer, steps *st, int chunk_size, double *max_delta, int rank, int size) {
    double new_phi;
    // Нижняя грань пространства;
    if (rank != RANK_ROOT) {
        // printf("rank - %d\n", rank);
        // Верхнего значения может не быть, нужно брать из верхнего буфера
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                // printf("first mul = %lf\n",(1.0 / ((2.0 / (st->x * st->x)) + (2.0 / (st->y * st->y)) + (2.0 / (st->z * st->z)) + ALPHA)));
                // printf("%lf, %lf, %lf, %lf, %lf, %lf\n", subspace[j * Nx + i + 1], subspace[j * Nx + i - 1], subspace[(j + 1) * Nx + i], subspace[(j - 1) * Nx + i], lower_buffer[(j - 1) * (Nx - 2) + i - 1], subspace[Nx * Ny + j * Nx + i]);
                new_phi = (1.0 / ((2.0 / (st->x * st->x)) + (2.0 / (st->y * st->y)) + (2.0 / (st->z * st->z)) + ALPHA))
                * (((subspace[j * Nx + i + 1] + subspace[j * Nx + i - 1]) / (st->x * st->x)) 
                + ((subspace[(j + 1) * Nx + i] + subspace[(j - 1) * Nx + i]) / (st->y * st->y)) 
                + ((lower_buffer[(j - 1) * (Nx - 2) + i - 1] + subspace[Nx * Ny + j * Nx + i]) / (st->z * st->z)) - func_ro(x0 + i * st->x, y0 + j * st->y, z0 + st->offset_z * st->z));    
                buffer_space[j * Nx + i] = new_phi;
                // printf("1: prev - %lf, cur - %lf\n", subspace[j * Nx + i], new_phi);
                calc_max_delta(subspace[j * Nx + i], new_phi, max_delta);
            }
        }
        MPI_Isend(&buffer_space[Nx + 1], 1, *inner_layer, rank - 1, 123, MPI_COMM_WORLD, send_lower_buff);
    }

    // Верхняя грань пространства
    if (rank != (size - 1)) {
        // Здесь может не быть нижнего буфера, если chunk_size == 1
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                new_phi = (1.0 / ((2.0 / (st->x * st->x)) + (2.0 / (st->y * st->y)) + (2.0 / (st->z * st->z)) + ALPHA))
                * (((subspace[(chunk_size - 1) * Nx * Ny + j * Nx + i + 1] + subspace[(chunk_size - 1) * Nx * Ny + j * Nx + i - 1]) / (st->x * st->x)) 
                + ((subspace[(chunk_size - 1) * Nx * Ny + (j + 1)* Nx + i] + subspace[(chunk_size - 1) * Nx * Ny + (j - 1) * Nx + i]) / (st->y * st->y)) 
                + ((upper_buffer[(j - 1) * (Nx - 2) + i - 1] + subspace[(chunk_size - 2) * Nx * Ny + j * Nx + i]) / (st->z * st->z)) - func_ro(x0 + i * st->x, y0 + j * st->y, z0 + (st->offset_z + chunk_size) * st->z));
                buffer_space[(chunk_size - 1) * Ny * Nx + j * Nx + i] = new_phi;
                // printf("0: prev - %lf, cur - %lf\n", subspace[(chunk_size - 1) * Nx * Ny + j * Nx + i], new_phi);

                calc_max_delta(subspace[(chunk_size - 1) * Ny * Nx + j * Nx + i], new_phi, max_delta);
            }
        }
        MPI_Isend(&buffer_space[(chunk_size - 1) * Nx * Ny + Nx + 1], 1, *inner_layer, rank + 1, 123, MPI_COMM_WORLD, send_upper_buff);
    }
}


void calc_inners(double *subspace, double *buffer_space, steps *st, int chunk_size, double *max_delta, int rank, int size) {
    double new_phi;
    for (int k = 1; k < chunk_size - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                //printf("%lf, %lf, %lf, %lf, %lf, %lf\n", subspace[j * Nx + i + 1], subspace[j * Nx + i - 1], subspace[(j + 1) * Nx + i], subspace[(j - 1) * Nx + i], subspace[(k + 1) * Nx * Ny + j * Nx + i], subspace[Nx * Ny + j * Nx + i]);
                new_phi = (1.0 / ((2.0 / (st->x * st->x)) + (2.0 / (st->y * st->y)) + (2.0 / (st->z * st->z)) + ALPHA))
                * (((subspace[k * Nx * Ny + j * Nx + i + 1] + subspace[k * Nx * Ny + j * Nx + i - 1]) / (st->x * st->x)) 
                + ((subspace[k * Nx * Ny + (j + 1)* Nx + i] + subspace[k * Nx * Ny + (j - 1) * Nx + i]) / (st->y * st->y)) 
                + ((subspace[(k + 1) * Nx * Ny + j * Nx + i] + subspace[(k - 1) * Nx * Ny + j * Nx + i]) / (st->z * st->z)) - func_ro(x0 + i * st->x, y0 + j * st->y, z0 + (k + st->offset_z) * st->z));            
                buffer_space[k * Nx * Ny + j * Nx + i] = new_phi; 
                calc_max_delta(subspace[k * Ny * Nx + j * Nx + i], new_phi, max_delta);
            }
        }
    }
}


void recv_buffers(MPI_Request *recv_lower_buff, MPI_Request *recv_upper_buff, double *lower_buffer, double *upper_buffer, int rank, int size) {
    // printf("Start recv - %d\n", rank);
    if (rank != RANK_ROOT) {
        MPI_Irecv(lower_buffer, (Nx - 2) * (Ny - 2), MPI_DOUBLE, rank - 1, 123, MPI_COMM_WORLD, recv_lower_buff);
        // MPI_Wait(upper_request, MPI_STATUS_IGNORE);
    }

    if (rank != size - 1) {
        MPI_Irecv(upper_buffer, (Nx - 2) * (Ny - 2), MPI_DOUBLE, rank + 1, 123, MPI_COMM_WORLD, recv_upper_buff);
        // MPI_Wait(lower_request, MPI_STATUS_IGNORE);
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
        upper_buffer = malloc(sizeof(double) * (Nx - 2) * (Ny - 2));
    }
    else if (rank == size - 1) {
        lower_buffer = malloc(sizeof(double) * (Nx - 2) * (Ny - 2));
    }
    else {
        upper_buffer = malloc(sizeof(double) * (Nx - 2) * (Ny - 2));
        lower_buffer = malloc(sizeof(double) * (Nx - 2) * (Ny - 2));
    }
    double *prev_space = malloc(sizeof(double) * Nx * Ny * chunk_size);
    double *cur_space = malloc(sizeof(double) * Nx * Ny * chunk_size);
    fill_borders(prev_space, chunk_size, &st);
    // if (rank == RANK_ROOT) {
    //     print_space(subspace, chunk_size);
    // }
    fill_borders(cur_space, chunk_size, &st);
    fill_inner(prev_space, chunk_size);

    init_buffers(lower_buffer, upper_buffer, rank, size);

    MPI_Request send_upper_buff;
    MPI_Request send_lower_buff;
    MPI_Request recv_lower_buff;
    MPI_Request recv_upper_buff;


    MPI_Datatype inner_layer;
    MPI_Type_vector(Ny - 2, Nx - 2, Nx, MPI_DOUBLE, &inner_layer);
    MPI_Type_commit(&inner_layer);

    int iterations = 999999;
    while (max_delta > EPSILON && iterations > 0) {
        local_delta = 0;
        iterations--;
        calc_borders(prev_space, cur_space, lower_buffer, upper_buffer, &send_lower_buff, &send_upper_buff, &inner_layer, &st, chunk_size, &local_delta, rank, size);
        recv_buffers(&recv_lower_buff, &recv_upper_buff, lower_buffer, upper_buffer, rank, size); // Прием ассинхронных запросов

        //printf("local delta - %lf\n", local_delta);
        calc_inners(prev_space, cur_space, &st, chunk_size, &local_delta, rank, size);
        //printf("local delta - %lf\n", local_delta);

        MPI_Allreduce(&local_delta, &max_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == RANK_ROOT)
            printf("accuracy - %.10lf\n", max_delta);
        
        // print_space(subspace, chunk_size);
        swap(&cur_space, &prev_space); 
        wait_buffer_swap(&send_lower_buff, &send_upper_buff, &recv_lower_buff, &recv_upper_buff, rank, size);
    }
    local_delta = 0;
    for (int k = 0; k < chunk_size; k++) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; i++) {
                calc_max_delta(prev_space[k * Nx * Ny + j * Nx + i], func_phi(x0 + i * st.x, y0 + j * st.y, z0 + (k + st.offset_z) * st.z), &local_delta);
            }
        }
    }
    // printf("proc - %d, delta - %lf\n", rank, local_delta);


    MPI_Allreduce(&local_delta, &max_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == RANK_ROOT)
        printf("Delta - %.10lf\n", max_delta);
    MPI_Type_free(&inner_layer);
    MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == RANK_ROOT) {
    //     print_space(subspace, chunk_size);
    //     // for (int k = 0; k < chunk_size; k++) {
    //     //     printf("layer %d\n", k);
    //     //     for (int j = 0; j < Ny; ++j) {
    //     //         for (int i = 0; i < Nx; i++) {
    //     //             printf("%lf ", func_phi(x0 + i * st.x, y0 + j * st.y, z0 + (k + st.offset_z) * st.z));
    //     //         }
    //     //         printf("\n");
    //     //     }
    //     // }
    // }
    // printf("rank - %d\n", rank);

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