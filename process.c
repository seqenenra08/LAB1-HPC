#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <string.h>

/* ---------------------------------------------------------------
   Memoria compartida: se usa mmap con MAP_SHARED | MAP_ANONYMOUS
   para que los procesos hijo escriban directamente en C.
   A y B son de solo lectura, así que se mantienen como arreglos
   planos en memoria compartida también (evitar cow innecesario).
--------------------------------------------------------------- */

static double *shm_alloc(size_t bytes) {
    void *p = mmap(NULL, bytes,
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(EXIT_FAILURE); }
    return (double *)p;
}

/* Acceso fila-columna sobre arreglo plano */
#define IDX(n, i, j)  ((i)*(n) + (j))

/* ---------------------------------------------------------------
   Inicializar A y B (arreglos planos)
--------------------------------------------------------------- */
static void inicializar(double *M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (double)(rand() % 100) / 10.0;
}

/* ---------------------------------------------------------------
   Multiplicar filas [row_ini, row_fin) de C = A x B
--------------------------------------------------------------- */
static void multiplicar_rango(const double *A, const double *B, double *C,
                               int n, int row_ini, int row_fin) {
    for (int i = row_ini; i < row_fin; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int k = 0; k < n; k++)
                acc += A[IDX(n,i,k)] * B[IDX(n,k,j)];
            C[IDX(n,i,j)] = acc;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Uso: %s <tamaño> [num_procesos]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n          = atoi(argv[1]);
    int num_procs  = (argc == 3) ? atoi(argv[2]) : (int)sysconf(_SC_NPROCESSORS_ONLN);

    if (n <= 0)         { fprintf(stderr, "Error: tamaño inválido.\n");        return EXIT_FAILURE; }
    if (num_procs <= 0) { fprintf(stderr, "Error: num_procesos inválido.\n");  return EXIT_FAILURE; }
    if (num_procs > n)    num_procs = n;   /* no tiene sentido más procesos que filas */

    srand(42);

    size_t bytes = (size_t)n * n * sizeof(double);

    /* Reservar A, B, C en memoria compartida */
    double *A = shm_alloc(bytes);
    double *B = shm_alloc(bytes);
    double *C = shm_alloc(bytes);

    inicializar(A, n);
    inicializar(B, n);

    /* ---- Lanzar procesos hijos ---- */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pid_t *pids = malloc(num_procs * sizeof(pid_t));

    for (int p = 0; p < num_procs; p++) {
        /* Reparto de filas: se distribuyen lo más equitativamente posible */
        int base      = n / num_procs;
        int remainder = n % num_procs;
        int row_ini   = p * base + (p < remainder ? p : remainder);
        int row_fin   = row_ini + base + (p < remainder ? 1 : 0);

        pids[p] = fork();
        if (pids[p] < 0) { perror("fork"); exit(EXIT_FAILURE); }

        if (pids[p] == 0) {               /* proceso hijo */
            multiplicar_rango(A, B, C, n, row_ini, row_fin);
            exit(EXIT_SUCCESS);
        }
    }

    /* Padre espera a todos los hijos */
    for (int p = 0; p < num_procs; p++) {
        int status;
        waitpid(pids[p], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            fprintf(stderr, "Advertencia: hijo %d terminó con error.\n", p);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double tiempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("Tiempo (wall-clock): %.6f s  |  n=%d  |  procesos=%d\n",
           tiempo, n, num_procs);

    free(pids);
    munmap(A, bytes);
    munmap(B, bytes);
    munmap(C, bytes);

    return EXIT_SUCCESS;
}