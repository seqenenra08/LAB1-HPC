#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>   /* gettimeofday – disponible sin macros extra */
#include <pthread.h>

/* ── Contexto global compartido entre hilos ── */
typedef struct {
    double **A;
    double **B;
    double **C;
    int     n;
    int     n_threads;
    int     thread_id;
} ThreadArgs;

/* ── Asignación / liberación de matrices ── */
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

/* ── Función ejecutada por cada hilo ──
   Cada hilo calcula un bloque de filas de C:
     filas [thread_id * chunk  ..  (thread_id+1) * chunk - 1]
   El último hilo absorbe las filas sobrantes.                */
void *multiplicar_bloque(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;
    int chunk = a->n / a->n_threads;
    int fila_ini = a->thread_id * chunk;
    int fila_fin = (a->thread_id == a->n_threads - 1)
                   ? a->n                       /* último hilo toma el resto */
                   : fila_ini + chunk;

    for (int i = fila_ini; i < fila_fin; i++) {
        for (int j = 0; j < a->n; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->n; k++)
                sum += a->A[i][k] * a->B[k][j];
            a->C[i][j] = sum;
        }
    }
    return NULL;
}

/* ── main ── */
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <tamaño> <hilos>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n        = atoi(argv[1]);
    int n_threads = atoi(argv[2]);

    if (n <= 0 || n_threads <= 0) {
        fprintf(stderr, "Error: argumentos inválidos.\n");
        return EXIT_FAILURE;
    }
    if (n_threads > n) n_threads = n;   /* no más hilos que filas */

    srand(42);

    double **A = crear_matriz(n);
    double **B = crear_matriz(n);
    double **C = crear_matriz(n);

    inicializar_matriz(A, n);
    inicializar_matriz(B, n);

    /* ── Crear hilos ── */
    pthread_t  *tids = (pthread_t  *)malloc(n_threads * sizeof(pthread_t));
    ThreadArgs *args = (ThreadArgs *)malloc(n_threads * sizeof(ThreadArgs));

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int t = 0; t < n_threads; t++) {
        args[t].A         = A;
        args[t].B         = B;
        args[t].C         = C;
        args[t].n         = n;
        args[t].n_threads = n_threads;
        args[t].thread_id = t;
        pthread_create(&tids[t], NULL, multiplicar_bloque, &args[t]);
    }

    for (int t = 0; t < n_threads; t++)
        pthread_join(tids[t], NULL);

    gettimeofday(&t1, NULL);

    double tiempo = (t1.tv_sec  - t0.tv_sec)
                  + (t1.tv_usec - t0.tv_usec) * 1e-6;

    /* Formato: hilos  tamaño  tiempo(s)  — coincide con el encabezado del script */
    printf("%d %d %.6f\n", n_threads, n, tiempo);

    free(tids);
    free(args);
    liberar_matriz(A, n);
    liberar_matriz(B, n);
    liberar_matriz(C, n);

    return EXIT_SUCCESS;
}