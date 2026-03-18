#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
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