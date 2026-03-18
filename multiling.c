#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 64

double **crear_matriz(int n) {
    double **M = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        M[i] = (double *)malloc(n * sizeof(double));
    return M;
}

void liberar_matriz(double **M, int n) {
    for (int i = 0; i < n; i++)
        free(M[i]);
    free(M);
}

void inicializar_matriz(double **M, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = (double)(rand() % 100) / 10.0;
}

void multiplicar_matrices(double **A, double **B, double **C, int n) {
    // Inicializar C en cero
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;

    for (int ii = 0; ii < n; ii += TILE_SIZE) {
        for (int kk = 0; kk < n; kk += TILE_SIZE) {
            for (int jj = 0; jj < n; jj += TILE_SIZE) {

                int i_max = ii + TILE_SIZE < n ? ii + TILE_SIZE : n;
                int k_max = kk + TILE_SIZE < n ? kk + TILE_SIZE : n;
                int j_max = jj + TILE_SIZE < n ? jj + TILE_SIZE : n;

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_ik = A[i][k]; // Registro: evita recargar A[i][k]
                        for (int j = jj; j < j_max; j++) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }

            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s <tamaño>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Error: tamaño inválido.\n");
        return EXIT_FAILURE;
    }

    srand(42);

    double **A = crear_matriz(n);
    double **B = crear_matriz(n);
    double **C = crear_matriz(n);

    inicializar_matriz(A, n);
    inicializar_matriz(B, n);

    clock_t inicio = clock();
    multiplicar_matrices(A, B, C, n);
    clock_t fin = clock();

    double tiempo_cpu = (double)(fin - inicio) / CLOCKS_PER_SEC;

    printf("%.6f\n", tiempo_cpu);

    liberar_matriz(A, n);
    liberar_matriz(B, n);
    liberar_matriz(C, n);

    return EXIT_SUCCESS;
}