/*
 * jacobi_poisson_1d.c
 *
 * Implementación serial del método de Jacobi para la ecuación de Poisson 1D:
 *
 *   -u''(x) = f(x),  x in [0,1]
 *   u(0) = 0, u(1) = 0
 *
 * Caso de prueba tomado del PDF de Burkardt:
 *   f(x) = -x * (x + 3) * exp(x)
 *   u_exact(x) = x * (x - 1) * exp(x)
 *
 * Discretización:
 *   (-u_{i-1} + 2u_i - u_{i+1}) / h^2 = f(x_i)
 *
 * Actualización Jacobi:
 *   u_new[i] = (u_old[i-1] + u_old[i+1] + h^2 * f(x_i)) / 2
 *
 * Entradas recomendadas para el proyecto:
 *   1) k   -> define n = 2^k + 1
 *   2) tol -> tolerancia RMS del residual
 *
 * Modos de uso:
 *   ./jacobi_poisson_1d solve k tol max_iter
 *   ./jacobi_poisson_1d sweep k_min k_max tol max_iter
 *
 * Compilación:
 *   gcc -O3 -std=c11 jacobi_poisson_1d.c -lm -o jacobi_poisson_1d
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    int n;
    int k;
    int iterations;
    int converged;
    double h;
    double residual_rms;
    double error_rms;
    double elapsed_seconds;
    double *x;
    double *u;
} JacobiResult;

/* ============================
   Utilidades matemáticas
   ============================ */

static double forcing_term(double x) {
    return -x * (x + 3.0) * exp(x);
}

static double exact_solution(double x) {
    return x * (x - 1.0) * exp(x);
}

static int nodes_from_k(int k) {
    if (k < 0 || k > 29) {
        return -1;
    }
    return (1 << k) + 1;
}

static double wall_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ============================
   Cálculo de métricas
   ============================ */

static double compute_residual_rms(const double *u, const double *x, int n, double h) {
    double sumsq = 0.0;
    double h2 = h * h;

    /* Residual en frontera: u(0)=0, u(1)=0 */
    {
        double r0 = u[0] - 0.0;
        double rn = u[n - 1] - 0.0;
        sumsq += r0 * r0 + rn * rn;
    }

    /* Residual interior: (-u_{i-1} + 2u_i - u_{i+1})/h^2 - f(x_i) */
    for (int i = 1; i < n - 1; ++i) {
        double Au_minus_f = (-u[i - 1] + 2.0 * u[i] - u[i + 1]) / h2 - forcing_term(x[i]);
        sumsq += Au_minus_f * Au_minus_f;
    }

    return sqrt(sumsq / (double)n);
}

static double compute_error_rms(const double *u, const double *x, int n) {
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = u[i] - exact_solution(x[i]);
        sumsq += e * e;
    }
    return sqrt(sumsq / (double)n);
}

/* ============================
   Solver Jacobi
   ============================ */

static int jacobi_poisson_1d(int k, double tol, int max_iter, JacobiResult *result) {
    if (!result) return 0;

    int n = nodes_from_k(k);
    if (n < 3) {
        fprintf(stderr, "Error: k invalido. Debe cumplir 0 <= k <= 29.\n");
        return 0;
    }
    if (tol <= 0.0) {
        fprintf(stderr, "Error: tol debe ser > 0.\n");
        return 0;
    }
    if (max_iter <= 0) {
        fprintf(stderr, "Error: max_iter debe ser > 0.\n");
        return 0;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *u_old = (double *)calloc((size_t)n, sizeof(double));
    double *u_new = (double *)calloc((size_t)n, sizeof(double));

    if (!x || !u_old || !u_new) {
        fprintf(stderr, "Error: no se pudo reservar memoria.\n");
        free(x);
        free(u_old);
        free(u_new);
        return 0;
    }

    double a = 0.0;
    double b = 1.0;
    double h = (b - a) / (double)(n - 1);
    double h2 = h * h;

    for (int i = 0; i < n; ++i) {
        x[i] = a + i * h;
    }

    /* Condiciones de frontera */
    u_old[0] = 0.0;
    u_old[n - 1] = 0.0;
    u_new[0] = 0.0;
    u_new[n - 1] = 0.0;

    double t0 = wall_time_seconds();

    int converged = 0;
    int iter = 0;
    double residual_rms = compute_residual_rms(u_old, x, n, h);

    while (iter < max_iter) {
        /* Paso Jacobi en puntos interiores */
        for (int i = 1; i < n - 1; ++i) {
            u_new[i] = 0.5 * (u_old[i - 1] + u_old[i + 1] + h2 * forcing_term(x[i]));
        }

        /* Reimponer frontera */
        u_new[0] = 0.0;
        u_new[n - 1] = 0.0;

        residual_rms = compute_residual_rms(u_new, x, n, h);
        iter++;

        if (residual_rms <= tol) {
            converged = 1;
            break;
        }

        /* swap punteros */
        double *tmp = u_old;
        u_old = u_new;
        u_new = tmp;
    }

    double elapsed = wall_time_seconds() - t0;

    /* Si convergió, la solución válida está en u_new.
       Si no convergió y salió por max_iter, depende del swap:
       - si acabó justo después de calcular u_new, u_new es la más reciente.
    */
    double *u_final = u_new;
    double error_rms = compute_error_rms(u_final, x, n);

    result->n = n;
    result->k = k;
    result->iterations = iter;
    result->converged = converged;
    result->h = h;
    result->residual_rms = residual_rms;
    result->error_rms = error_rms;
    result->elapsed_seconds = elapsed;
    result->x = x;
    result->u = u_final;

    /* Liberar solo el buffer que NO se devuelve */
    if (u_old == u_final) {
        free(u_new);
    } else {
        free(u_old);
    }

    return 1;
}

static void free_result(JacobiResult *result) {
    if (!result) return;
    free(result->x);
    free(result->u);
    result->x = NULL;
    result->u = NULL;
}

/* ============================
   Impresión
   ============================ */

static void print_solution_table(const JacobiResult *res) {
    printf("\n%-6s %-14s %-18s %-18s %-18s\n", "i", "x", "u_jacobi", "u_exact", "abs_error");
    printf("------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < res->n; ++i) {
        double ue = exact_solution(res->x[i]);
        double ae = fabs(res->u[i] - ue);
        printf("%-6d %-14.8f %-18.10e %-18.10e %-18.10e\n",
               i, res->x[i], res->u[i], ue, ae);
    }
}

static void print_summary(const JacobiResult *res) {
    printf("===== Resultado Jacobi Poisson 1D =====\n");
    printf("k               : %d\n", res->k);
    printf("n = 2^k + 1     : %d\n", res->n);
    printf("h               : %.10e\n", res->h);
    printf("iteraciones     : %d\n", res->iterations);
    printf("convergio       : %s\n", res->converged ? "SI" : "NO");
    printf("residual RMS    : %.10e\n", res->residual_rms);
    printf("error RMS       : %.10e\n", res->error_rms);
    printf("tiempo (s)      : %.10f\n", res->elapsed_seconds);
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Uso:\n"
        "  %s solve k tol max_iter\n"
        "  %s sweep k_min k_max tol max_iter\n\n"
        "Ejemplos:\n"
        "  %s solve 5 1e-6 100000\n"
        "  %s sweep 5 14 1e-6 1000000\n",
        prog, prog, prog, prog);
}

/* ============================
   Main
   ============================ */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "solve") == 0) {
        if (argc != 5) {
            print_usage(argv[0]);
            return 1;
        }

        int k = atoi(argv[2]);
        double tol = atof(argv[3]);
        int max_iter = atoi(argv[4]);

        JacobiResult res = {0};
        if (!jacobi_poisson_1d(k, tol, max_iter, &res)) {
            return 1;
        }

        print_summary(&res);

        /* Imprime la tabla completa solo si no es demasiado grande */
        if (res.n <= 65) {
            print_solution_table(&res);
        } else {
            printf("\nTabla de solucion omitida porque n=%d es grande.\n", res.n);
        }

        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "sweep") == 0) {
        if (argc != 6) {
            print_usage(argv[0]);
            return 1;
        }

        int k_min = atoi(argv[2]);
        int k_max = atoi(argv[3]);
        double tol = atof(argv[4]);
        int max_iter = atoi(argv[5]);

        if (k_min > k_max) {
            fprintf(stderr, "Error: k_min debe ser <= k_max.\n");
            return 1;
        }

        printf("k,n,h,iterations,converged,residual_rms,error_rms,time_seconds\n");

        for (int k = k_min; k <= k_max; ++k) {
            JacobiResult res = {0};
            if (!jacobi_poisson_1d(k, tol, max_iter, &res)) {
                fprintf(stderr, "Error al resolver para k=%d\n", k);
                continue;
            }

            printf("%d,%d,%.10e,%d,%d,%.10e,%.10e,%.10f\n",
                   res.k, res.n, res.h, res.iterations, res.converged,
                   res.residual_rms, res.error_rms, res.elapsed_seconds);

            free_result(&res);
        }

        return 0;
    }

    print_usage(argv[0]);
    return 1;
}