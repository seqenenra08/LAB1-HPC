#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    int n;
    int k;
    int iterations;
    int converged;
    int num_threads;
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

static double compute_residual_rms(const double *u, const double *x, int n, double h) {
    double sumsq = 0.0;
    double h2 = h * h;

    {
        double r0 = u[0] - 0.0;
        double rn = u[n - 1] - 0.0;
        sumsq += r0 * r0 + rn * rn;
    }

    for (int i = 1; i < n - 1; ++i) {
        double r = (-u[i - 1] + 2.0 * u[i] - u[i + 1]) / h2 - forcing_term(x[i]);
        sumsq += r * r;
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
   Solver serial
   ============================ */

static int jacobi_poisson_1d_serial(int k, double tol, int max_iter, JacobiResult *result) {
    if (!result) return 0;

    int n = nodes_from_k(k);
    if (n < 3) {
        fprintf(stderr, "Error: k invalido.\n");
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

    u_old[0] = 0.0;
    u_old[n - 1] = 0.0;
    u_new[0] = 0.0;
    u_new[n - 1] = 0.0;

    double t0 = wall_time_seconds();

    int converged = 0;
    int iter = 0;
    double residual_rms = compute_residual_rms(u_old, x, n, h);

    if (residual_rms <= tol) {
        converged = 1;
    } else {
        while (iter < max_iter) {
            for (int i = 1; i < n - 1; ++i) {
                u_new[i] = 0.5 * (u_old[i - 1] + u_old[i + 1] + h2 * forcing_term(x[i]));
            }

            u_new[0] = 0.0;
            u_new[n - 1] = 0.0;

            residual_rms = compute_residual_rms(u_new, x, n, h);
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

    double *u_final = converged ? u_new : (iter == 0 ? u_old : u_new);
    double error_rms = compute_error_rms(u_final, x, n);

    result->n = n;
    result->k = k;
    result->iterations = iter;
    result->converged = converged;
    result->num_threads = 1;
    result->h = h;
    result->residual_rms = residual_rms;
    result->error_rms = error_rms;
    result->elapsed_seconds = elapsed;
    result->x = x;
    result->u = u_final;

    if (u_final == u_old) {
        free(u_new);
    } else {
        free(u_old);
    }

    return 1;
}

/* ============================
   Solver con POSIX threads
   ============================ */

typedef struct {
    int n;
    int k;
    int max_iter;
    int num_threads;
    int iterations;
    int stop;
    int converged;
    double tol;
    double h;
    double h2;
    double residual_rms;
    double *x;
    double *u_a;
    double *u_b;
    double *u_old;
    double *u_new;
    pthread_barrier_t barrier;
} SharedData;

typedef struct {
    int tid;
    int start;
    int end;
    SharedData *shared;
} ThreadTask;

static void *jacobi_worker(void *arg) {
    ThreadTask *task = (ThreadTask *)arg;
    SharedData *s = task->shared;

    while (1) {
        for (int i = task->start; i <= task->end; ++i) {
            s->u_new[i] = 0.5 * (s->u_old[i - 1] + s->u_old[i + 1] + s->h2 * forcing_term(s->x[i]));
        }

        if (task->tid == 0) {
            s->u_new[0] = 0.0;
            s->u_new[s->n - 1] = 0.0;
        }

        pthread_barrier_wait(&s->barrier);

        if (task->tid == 0) {
            s->residual_rms = compute_residual_rms(s->u_new, s->x, s->n, s->h);
            s->iterations++;

            if (s->residual_rms <= s->tol) {
                s->converged = 1;
                s->stop = 1;
            } else if (s->iterations >= s->max_iter) {
                s->converged = 0;
                s->stop = 1;
            } else {
                double *tmp = s->u_old;
                s->u_old = s->u_new;
                s->u_new = tmp;
            }
        }

        pthread_barrier_wait(&s->barrier);

        if (s->stop) {
            break;
        }
    }

    return NULL;
}

static int jacobi_poisson_1d_threads(int k, double tol, int max_iter, int num_threads, JacobiResult *result) {
    if (!result) return 0;

    int n = nodes_from_k(k);
    if (n < 3) {
        fprintf(stderr, "Error: k invalido.\n");
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
    if (num_threads <= 0) {
        fprintf(stderr, "Error: num_threads debe ser > 0.\n");
        return 0;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *u_a = (double *)calloc((size_t)n, sizeof(double));
    double *u_b = (double *)calloc((size_t)n, sizeof(double));
    pthread_t *threads = (pthread_t *)malloc((size_t)num_threads * sizeof(pthread_t));
    ThreadTask *tasks = (ThreadTask *)malloc((size_t)num_threads * sizeof(ThreadTask));

    if (!x || !u_a || !u_b || !threads || !tasks) {
        fprintf(stderr, "Error: no se pudo reservar memoria.\n");
        free(x);
        free(u_a);
        free(u_b);
        free(threads);
        free(tasks);
        return 0;
    }

    double a = 0.0;
    double b = 1.0;
    double h = (b - a) / (double)(n - 1);

    for (int i = 0; i < n; ++i) {
        x[i] = a + i * h;
    }

    u_a[0] = 0.0;
    u_a[n - 1] = 0.0;
    u_b[0] = 0.0;
    u_b[n - 1] = 0.0;

    SharedData s;
    s.n = n;
    s.k = k;
    s.max_iter = max_iter;
    s.num_threads = num_threads;
    s.iterations = 0;
    s.stop = 0;
    s.converged = 0;
    s.tol = tol;
    s.h = h;
    s.h2 = h * h;
    s.x = x;
    s.u_a = u_a;
    s.u_b = u_b;
    s.u_old = u_a;
    s.u_new = u_b;
    s.residual_rms = compute_residual_rms(s.u_old, s.x, s.n, s.h);

    double t0 = wall_time_seconds();

    if (s.residual_rms <= tol) {
        s.converged = 1;
    } else {
        if (pthread_barrier_init(&s.barrier, NULL, (unsigned)num_threads) != 0) {
            fprintf(stderr, "Error: no se pudo inicializar la barrera pthread.\n");
            free(x);
            free(u_a);
            free(u_b);
            free(threads);
            free(tasks);
            return 0;
        }

        int interior = n - 2;
        int base = interior / num_threads;
        int extra = interior % num_threads;
        int current = 1;

        for (int t = 0; t < num_threads; ++t) {
            int count = base + (t < extra ? 1 : 0);

            tasks[t].tid = t;
            tasks[t].start = current;
            tasks[t].end = current + count - 1;
            tasks[t].shared = &s;

            current += count;

            if (pthread_create(&threads[t], NULL, jacobi_worker, &tasks[t]) != 0) {
                fprintf(stderr, "Error: no se pudo crear el hilo %d.\n", t);

                s.stop = 1;
                for (int j = 0; j < t; ++j) {
                    pthread_join(threads[j], NULL);
                }
                pthread_barrier_destroy(&s.barrier);
                free(x);
                free(u_a);
                free(u_b);
                free(threads);
                free(tasks);
                return 0;
            }
        }

        for (int t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], NULL);
        }

        pthread_barrier_destroy(&s.barrier);
    }

    double elapsed = wall_time_seconds() - t0;

    double *u_final;
    if (s.iterations == 0) {
        u_final = s.u_old;
    } else {
        u_final = s.u_new;
    }

    double error_rms = compute_error_rms(u_final, x, n);

    result->n = n;
    result->k = k;
    result->iterations = s.iterations;
    result->converged = s.converged;
    result->num_threads = num_threads;
    result->h = h;
    result->residual_rms = s.residual_rms;
    result->error_rms = error_rms;
    result->elapsed_seconds = elapsed;
    result->x = x;
    result->u = u_final;

    if (u_final == u_a) {
        free(u_b);
    } else {
        free(u_a);
    }

    free(threads);
    free(tasks);

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

static void print_summary_serial(const JacobiResult *res) {
    printf("===== Resultado Jacobi Poisson 1D (Serial) =====\n");
    printf("k               : %d\n", res->k);
    printf("n = 2^k + 1     : %d\n", res->n);
    printf("h               : %.10e\n", res->h);
    printf("iteraciones     : %d\n", res->iterations);
    printf("convergio       : %s\n", res->converged ? "SI" : "NO");
    printf("residual RMS    : %.10e\n", res->residual_rms);
    printf("error RMS       : %.10e\n", res->error_rms);
    printf("tiempo (s)      : %.10f\n", res->elapsed_seconds);
}

static void print_summary_threads(const JacobiResult *res) {
    printf("===== Resultado Jacobi Poisson 1D (POSIX Threads) =====\n");
    printf("k               : %d\n", res->k);
    printf("threads         : %d\n", res->num_threads);
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
           res->num_threads,
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
        "  %s solve_threads k tol max_iter num_threads\n"
        "  %s csv_serial k tol max_iter\n"
        "  %s csv_threads k tol max_iter num_threads\n\n"
        "Ejemplos:\n"
        "  %s solve 5 1e-6 1000000\n"
        "  %s solve_threads 12 1e-6 1000000 6\n"
        "  %s csv_serial 12 1e-6 1000000\n"
        "  %s csv_threads 12 1e-6 1000000 6\n",
        prog, prog, prog, prog,
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
        if (!jacobi_poisson_1d_serial(k, tol, max_iter, &res)) {
            return 1;
        }

        print_summary_serial(&res);

        if (res.n <= 65) {
            print_solution_table(&res);
        } else {
            printf("\nTabla de solucion omitida porque n=%d es grande.\n", res.n);
        }

        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "solve_threads") == 0) {
        if (argc != 6) {
            print_usage(argv[0]);
            return 1;
        }

        int k = atoi(argv[2]);
        double tol = atof(argv[3]);
        int max_iter = atoi(argv[4]);
        int num_threads = atoi(argv[5]);

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_threads(k, tol, max_iter, num_threads, &res)) {
            return 1;
        }

        print_summary_threads(&res);

        if (res.n <= 65) {
            print_solution_table(&res);
        } else {
            printf("\nTabla de solucion omitida porque n=%d es grande.\n", res.n);
        }

        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "csv_serial") == 0) {
        if (argc != 5) {
            print_usage(argv[0]);
            return 1;
        }

        int k = atoi(argv[2]);
        double tol = atof(argv[3]);
        int max_iter = atoi(argv[4]);

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_serial(k, tol, max_iter, &res)) {
            return 1;
        }

        print_csv_line("serial", &res);
        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "csv_threads") == 0) {
        if (argc != 6) {
            print_usage(argv[0]);
            return 1;
        }

        int k = atoi(argv[2]);
        double tol = atof(argv[3]);
        int max_iter = atoi(argv[4]);
        int num_threads = atoi(argv[5]);

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_threads(k, tol, max_iter, num_threads, &res)) {
            return 1;
        }

        print_csv_line("threads", &res);
        free_result(&res);
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}