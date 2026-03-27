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
    int block_size;
    double h;
    double residual_rms;
    double error_rms;
    double elapsed_seconds;
    double *x;
    double *u;
} JacobiResult;

/* ============================
   Problema de prueba
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
   Métricas
   ============================ */

static double compute_error_rms(const double *u, const double *x, int n) {
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = u[i] - exact_solution(x[i]);
        sumsq += e * e;
    }
    return sqrt(sumsq / (double)n);
}

/* Residual RMS usando f(x) ya precomputado */
static double compute_residual_rms_precomputed(const double *u,
                                               const double *fvec,
                                               int n,
                                               double h) {
    double sumsq = 0.0;
    double h2 = h * h;

    /* frontera */
    sumsq += u[0] * u[0];
    sumsq += u[n - 1] * u[n - 1];

    for (int i = 1; i < n - 1; ++i) {
        double r = (-u[i - 1] + 2.0 * u[i] - u[i + 1]) / h2 - fvec[i];
        sumsq += r * r;
    }

    return sqrt(sumsq / (double)n);
}

/* ============================
   Solver serial base
   ============================ */

static int jacobi_poisson_1d_serial(int k, double tol, int max_iter, JacobiResult *result) {
    if (!result) return 0;

    int n = nodes_from_k(k);
    if (n < 3 || tol <= 0.0 || max_iter <= 0) {
        return 0;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *fvec = (double *)malloc((size_t)n * sizeof(double));
    double *rhs = (double *)malloc((size_t)n * sizeof(double));
    double *u_old = (double *)calloc((size_t)n, sizeof(double));
    double *u_new = (double *)calloc((size_t)n, sizeof(double));

    if (!x || !fvec || !rhs || !u_old || !u_new) {
        free(x);
        free(fvec);
        free(rhs);
        free(u_old);
        free(u_new);
        return 0;
    }

    double h = 1.0 / (double)(n - 1);
    double h2 = h * h;

    for (int i = 0; i < n; ++i) {
        x[i] = i * h;
        fvec[i] = forcing_term(x[i]);
        rhs[i] = h2 * fvec[i];
    }

    u_old[0] = 0.0;
    u_old[n - 1] = 0.0;
    u_new[0] = 0.0;
    u_new[n - 1] = 0.0;

    int iter = 0;
    int converged = 0;
    double residual_rms = compute_residual_rms_precomputed(u_old, fvec, n, h);

    double t0 = wall_time_seconds();

    if (residual_rms <= tol) {
        converged = 1;
    } else {
        while (iter < max_iter) {
            for (int i = 1; i < n - 1; ++i) {
                u_new[i] = 0.5 * (u_old[i - 1] + u_old[i + 1] + rhs[i]);
            }

            u_new[0] = 0.0;
            u_new[n - 1] = 0.0;

            residual_rms = compute_residual_rms_precomputed(u_new, fvec, n, h);
            iter++;

            if (residual_rms <= tol) {
                converged = 1;
                break;
            }

            double *tmp = u_old;
            u_old = u_new;
            u_new = tmp;
        }
    }

    double elapsed = wall_time_seconds() - t0;
    double *u_final = (iter == 0) ? u_old : u_new;
    double error_rms = compute_error_rms(u_final, x, n);

    result->n = n;
    result->k = k;
    result->iterations = iter;
    result->converged = converged;
    result->block_size = 0;
    result->h = h;
    result->residual_rms = residual_rms;
    result->error_rms = error_rms;
    result->elapsed_seconds = elapsed;
    result->x = x;
    result->u = u_final;

    free(fvec);
    free(rhs);

    if (u_final == u_old) {
        free(u_new);
    } else {
        free(u_old);
    }

    return 1;
}

/* ============================
   Solver serial optimizado por memoria con tiling
   ============================ */

static int jacobi_poisson_1d_tiled(int k, double tol, int max_iter, int block_size, JacobiResult *result) {
    if (!result) return 0;

    int n = nodes_from_k(k);
    if (n < 3 || tol <= 0.0 || max_iter <= 0 || block_size <= 0) {
        return 0;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *fvec = (double *)malloc((size_t)n * sizeof(double));
    double *rhs = (double *)malloc((size_t)n * sizeof(double));
    double *u_old = (double *)calloc((size_t)n, sizeof(double));
    double *u_new = (double *)calloc((size_t)n, sizeof(double));

    if (!x || !fvec || !rhs || !u_old || !u_new) {
        free(x);
        free(fvec);
        free(rhs);
        free(u_old);
        free(u_new);
        return 0;
    }

    double h = 1.0 / (double)(n - 1);
    double h2 = h * h;

    /* Precomputo de x, f(x) y h^2*f(x) */
    for (int i = 0; i < n; ++i) {
        x[i] = i * h;
        fvec[i] = forcing_term(x[i]);
        rhs[i] = h2 * fvec[i];
    }

    u_old[0] = 0.0;
    u_old[n - 1] = 0.0;
    u_new[0] = 0.0;
    u_new[n - 1] = 0.0;

    int iter = 0;
    int converged = 0;
    double residual_rms = compute_residual_rms_precomputed(u_old, fvec, n, h);

    double t0 = wall_time_seconds();

    if (residual_rms <= tol) {
        converged = 1;
    } else {
        while (iter < max_iter) {

            /* ============================
               Paso Jacobi por bloques
               ============================ */
            for (int b = 1; b < n - 1; b += block_size) {
                int end = b + block_size;
                if (end > n - 1) {
                    end = n - 1;
                }

                for (int i = b; i < end; ++i) {
                    u_new[i] = 0.5 * (u_old[i - 1] + u_old[i + 1] + rhs[i]);
                }
            }

            u_new[0] = 0.0;
            u_new[n - 1] = 0.0;

            /* ============================
               Residual por bloques
               ============================ */
            {
                double sumsq = 0.0;

                sumsq += u_new[0] * u_new[0];
                sumsq += u_new[n - 1] * u_new[n - 1];

                for (int b = 1; b < n - 1; b += block_size) {
                    int end = b + block_size;
                    if (end > n - 1) {
                        end = n - 1;
                    }

                    for (int i = b; i < end; ++i) {
                        double r = (-u_new[i - 1] + 2.0 * u_new[i] - u_new[i + 1]) / h2 - fvec[i];
                        sumsq += r * r;
                    }
                }

                residual_rms = sqrt(sumsq / (double)n);
            }

            iter++;

            if (residual_rms <= tol) {
                converged = 1;
                break;
            }

            /* swap de punteros, evita copiar arreglos completos */
            double *tmp = u_old;
            u_old = u_new;
            u_new = tmp;
        }
    }

    double elapsed = wall_time_seconds() - t0;
    double *u_final = (iter == 0) ? u_old : u_new;
    double error_rms = compute_error_rms(u_final, x, n);

    result->n = n;
    result->k = k;
    result->iterations = iter;
    result->converged = converged;
    result->block_size = block_size;
    result->h = h;
    result->residual_rms = residual_rms;
    result->error_rms = error_rms;
    result->elapsed_seconds = elapsed;
    result->x = x;
    result->u = u_final;

    free(fvec);
    free(rhs);

    if (u_final == u_old) {
        free(u_new);
    } else {
        free(u_old);
    }

    return 1;
}

/* ============================
   Impresión
   ============================ */

static void free_result(JacobiResult *result) {
    if (!result) return;
    free(result->x);
    free(result->u);
    result->x = NULL;
    result->u = NULL;
}

static void print_summary_serial(const JacobiResult *res) {
    printf("===== Resultado Jacobi Poisson 1D (Serial Base) =====\n");
    printf("k               : %d\n", res->k);
    printf("n = 2^k + 1     : %d\n", res->n);
    printf("h               : %.10e\n", res->h);
    printf("iteraciones     : %d\n", res->iterations);
    printf("convergio       : %s\n", res->converged ? "SI" : "NO");
    printf("residual RMS    : %.10e\n", res->residual_rms);
    printf("error RMS       : %.10e\n", res->error_rms);
    printf("tiempo (s)      : %.10f\n", res->elapsed_seconds);
}

static void print_summary_tiled(const JacobiResult *res) {
    printf("===== Resultado Jacobi Poisson 1D (Serial + Tiling) =====\n");
    printf("k               : %d\n", res->k);
    printf("block_size      : %d\n", res->block_size);
    printf("n = 2^k + 1     : %d\n", res->n);
    printf("h               : %.10e\n", res->h);
    printf("iteraciones     : %d\n", res->iterations);
    printf("convergio       : %s\n", res->converged ? "SI" : "NO");
    printf("residual RMS    : %.10e\n", res->residual_rms);
    printf("error RMS       : %.10e\n", res->error_rms);
    printf("tiempo (s)      : %.10f\n", res->elapsed_seconds);
}

static void print_csv_line(const char *mode, const JacobiResult *res) {
    printf("%s,%d,%d,%d,%.10e,%d,%d,%.10e,%.10e,%.10f\n",
           mode,
           res->k,
           res->block_size,
           res->n,
           res->h,
           res->iterations,
           res->converged,
           res->residual_rms,
           res->error_rms,
           res->elapsed_seconds);
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Uso:\n"
        "  %s solve k tol max_iter\n"
        "  %s solve_tiled k tol max_iter block_size\n"
        "  %s csv_serial k tol max_iter\n"
        "  %s csv_tiled k tol max_iter block_size\n\n"
        "Ejemplos:\n"
        "  %s solve 8 1e-6 1000000\n"
        "  %s solve_tiled 8 1e-6 1000000 64\n"
        "  %s csv_serial 8 1e-6 1000000\n"
        "  %s csv_tiled 8 1e-6 1000000 64\n",
        prog, prog, prog, prog,
        prog, prog, prog, prog);
}

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

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_serial(atoi(argv[2]), atof(argv[3]), atoi(argv[4]), &res)) {
            fprintf(stderr, "Error en la ejecucion serial.\n");
            return 1;
        }

        print_summary_serial(&res);
        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "solve_tiled") == 0) {
        if (argc != 6) {
            print_usage(argv[0]);
            return 1;
        }

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_tiled(atoi(argv[2]), atof(argv[3]), atoi(argv[4]), atoi(argv[5]), &res)) {
            fprintf(stderr, "Error en la ejecucion tiled.\n");
            return 1;
        }

        print_summary_tiled(&res);
        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "csv_serial") == 0) {
        if (argc != 5) {
            print_usage(argv[0]);
            return 1;
        }

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_serial(atoi(argv[2]), atof(argv[3]), atoi(argv[4]), &res)) {
            return 1;
        }

        print_csv_line("serial_base", &res);
        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "csv_tiled") == 0) {
        if (argc != 6) {
            print_usage(argv[0]);
            return 1;
        }

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_tiled(atoi(argv[2]), atof(argv[3]), atoi(argv[4]), atoi(argv[5]), &res)) {
            return 1;
        }

        print_csv_line("serial_tiled", &res);
        free_result(&res);
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}