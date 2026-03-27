#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    int n;
    int k;
    int iterations;
    int converged;
    int num_processes;
    double h;
    double residual_rms;
    double error_rms;
    double elapsed_seconds;
    double *x;
    double *u;
} JacobiResult;

typedef struct {
    int n;
    int max_iter;
    int num_processes;
    int iterations;
    int stop;
    int converged;
    int current_old;
    double tol;
    double h;
    double h2;
    double residual_rms;
    sem_t done_sem;
} SharedControl;

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

static double compute_residual_rms(const double *u, const double *x, int n, double h) {
    double sumsq = 0.0;
    double h2 = h * h;

    sumsq += u[0] * u[0] + u[n - 1] * u[n - 1];

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

static int sem_wait_checked(sem_t *sem) {
    while (sem_wait(sem) == -1) {
        if (errno != EINTR) {
            return 0;
        }
    }
    return 1;
}

static double *alloc_shared_doubles(size_t count) {
    void *ptr = mmap(NULL, count * sizeof(double), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }
    return (double *)ptr;
}

static SharedControl *alloc_shared_control(void) {
    void *ptr = mmap(NULL, sizeof(SharedControl), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }
    return (SharedControl *)ptr;
}

static sem_t *alloc_shared_sems(size_t count) {
    void *ptr = mmap(NULL, count * sizeof(sem_t), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }
    return (sem_t *)ptr;
}

static int jacobi_poisson_1d_serial(int k, double tol, int max_iter, JacobiResult *result) {
    int n = nodes_from_k(k);
    if (!result || n < 3 || tol <= 0.0 || max_iter <= 0) {
        return 0;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *u_old = (double *)calloc((size_t)n, sizeof(double));
    double *u_new = (double *)calloc((size_t)n, sizeof(double));
    if (!x || !u_old || !u_new) {
        free(x);
        free(u_old);
        free(u_new);
        return 0;
    }

    double h = 1.0 / (double)(n - 1);
    double h2 = h * h;

    for (int i = 0; i < n; ++i) {
        x[i] = i * h;
    }

    double residual_rms = compute_residual_rms(u_old, x, n, h);
    int iter = 0;
    int converged = 0;

    double t0 = wall_time_seconds();

    if (residual_rms <= tol) {
        converged = 1;
    } else {
        while (iter < max_iter) {
            for (int i = 1; i < n - 1; ++i) {
                u_new[i] = 0.5 * (u_old[i - 1] + u_old[i + 1] + h2 * forcing_term(x[i]));
            }

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
    double *u_final = (iter == 0) ? u_old : u_new;
    double error_rms = compute_error_rms(u_final, x, n);

    result->n = n;
    result->k = k;
    result->iterations = iter;
    result->converged = converged;
    result->num_processes = 1;
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

static void worker_loop(SharedControl *ctrl, sem_t *my_start_sem,
                        const double *x, double *u_a, double *u_b,
                        int start, int end) {
    for (;;) {
        if (!sem_wait_checked(my_start_sem)) {
            _exit(1);
        }

        if (ctrl->stop) {
            break;
        }

        double *u_old = (ctrl->current_old == 0) ? u_a : u_b;
        double *u_new = (ctrl->current_old == 0) ? u_b : u_a;

        for (int i = start; i <= end; ++i) {
            u_new[i] = 0.5 * (u_old[i - 1] + u_old[i + 1] + ctrl->h2 * forcing_term(x[i]));
        }

        if (sem_post(&ctrl->done_sem) == -1) {
            _exit(1);
        }
    }

    _exit(0);
}

static int jacobi_poisson_1d_processes(int k, double tol, int max_iter, int num_processes,
                                       JacobiResult *result) {
    int n = nodes_from_k(k);
    if (!result || n < 3 || tol <= 0.0 || max_iter <= 0 || num_processes <= 0) {
        return 0;
    }

    double *x_sh = alloc_shared_doubles((size_t)n);
    double *u_a = alloc_shared_doubles((size_t)n);
    double *u_b = alloc_shared_doubles((size_t)n);
    SharedControl *ctrl = alloc_shared_control();
    sem_t *start_sems = alloc_shared_sems((size_t)num_processes);
    pid_t *pids = (pid_t *)calloc((size_t)num_processes, sizeof(pid_t));

    if (!x_sh || !u_a || !u_b || !ctrl || !start_sems || !pids) {
        if (x_sh) munmap(x_sh, (size_t)n * sizeof(double));
        if (u_a) munmap(u_a, (size_t)n * sizeof(double));
        if (u_b) munmap(u_b, (size_t)n * sizeof(double));
        if (ctrl) munmap(ctrl, sizeof(SharedControl));
        if (start_sems) munmap(start_sems, (size_t)num_processes * sizeof(sem_t));
        free(pids);
        return 0;
    }

    double h = 1.0 / (double)(n - 1);
    for (int i = 0; i < n; ++i) {
        x_sh[i] = i * h;
    }
    memset(u_a, 0, (size_t)n * sizeof(double));
    memset(u_b, 0, (size_t)n * sizeof(double));

    ctrl->n = n;
    ctrl->max_iter = max_iter;
    ctrl->num_processes = num_processes;
    ctrl->iterations = 0;
    ctrl->stop = 0;
    ctrl->converged = 0;
    ctrl->current_old = 0;
    ctrl->tol = tol;
    ctrl->h = h;
    ctrl->h2 = h * h;
    ctrl->residual_rms = compute_residual_rms(u_a, x_sh, n, h);

    if (sem_init(&ctrl->done_sem, 1, 0) == -1) {
        munmap(x_sh, (size_t)n * sizeof(double));
        munmap(u_a, (size_t)n * sizeof(double));
        munmap(u_b, (size_t)n * sizeof(double));
        munmap(ctrl, sizeof(SharedControl));
        munmap(start_sems, (size_t)num_processes * sizeof(sem_t));
        free(pids);
        return 0;
    }

    for (int p = 0; p < num_processes; ++p) {
        if (sem_init(&start_sems[p], 1, 0) == -1) {
            for (int j = 0; j < p; ++j) {
                sem_destroy(&start_sems[j]);
            }
            sem_destroy(&ctrl->done_sem);
            munmap(x_sh, (size_t)n * sizeof(double));
            munmap(u_a, (size_t)n * sizeof(double));
            munmap(u_b, (size_t)n * sizeof(double));
            munmap(ctrl, sizeof(SharedControl));
            munmap(start_sems, (size_t)num_processes * sizeof(sem_t));
            free(pids);
            return 0;
        }
    }

    double t0 = wall_time_seconds();

    int created = 0;
    int interior = n - 2;
    int base = interior / num_processes;
    int extra = interior % num_processes;
    int current = 1;

    for (int p = 0; p < num_processes; ++p) {
        int count = base + (p < extra ? 1 : 0);
        int start = current;
        int end = current + count - 1;
        current += count;

        pid_t pid = fork();
        if (pid < 0) {
            ctrl->stop = 1;
            for (int j = 0; j < created; ++j) {
                sem_post(&start_sems[j]);
            }
            for (int j = 0; j < created; ++j) {
                waitpid(pids[j], NULL, 0);
            }
            for (int j = 0; j < num_processes; ++j) {
                sem_destroy(&start_sems[j]);
            }
            sem_destroy(&ctrl->done_sem);
            munmap(x_sh, (size_t)n * sizeof(double));
            munmap(u_a, (size_t)n * sizeof(double));
            munmap(u_b, (size_t)n * sizeof(double));
            munmap(ctrl, sizeof(SharedControl));
            munmap(start_sems, (size_t)num_processes * sizeof(sem_t));
            free(pids);
            return 0;
        }

        if (pid == 0) {
            worker_loop(ctrl, &start_sems[p], x_sh, u_a, u_b, start, end);
        }

        pids[p] = pid;
        created++;
    }

    if (ctrl->residual_rms <= tol) {
        ctrl->converged = 1;
        ctrl->stop = 1;
    } else {
        while (ctrl->iterations < ctrl->max_iter) {
            for (int p = 0; p < num_processes; ++p) {
                if (sem_post(&start_sems[p]) == -1) {
                    ctrl->stop = 1;
                    break;
                }
            }
            if (ctrl->stop) {
                break;
            }

            for (int p = 0; p < num_processes; ++p) {
                if (!sem_wait_checked(&ctrl->done_sem)) {
                    ctrl->stop = 1;
                    break;
                }
            }
            if (ctrl->stop) {
                break;
            }

            int new_index = 1 - ctrl->current_old;
            double *u_new = (new_index == 0) ? u_a : u_b;

            ctrl->residual_rms = compute_residual_rms(u_new, x_sh, n, h);
            ctrl->iterations++;

            if (ctrl->residual_rms <= tol) {
                ctrl->converged = 1;
                ctrl->stop = 1;
                break;
            }

            if (ctrl->iterations >= ctrl->max_iter) {
                ctrl->converged = 0;
                ctrl->stop = 1;
                break;
            }

            ctrl->current_old = new_index;
        }
    }

    ctrl->stop = 1;
    for (int p = 0; p < num_processes; ++p) {
        sem_post(&start_sems[p]);
    }

    for (int p = 0; p < num_processes; ++p) {
        waitpid(pids[p], NULL, 0);
    }

    double elapsed = wall_time_seconds() - t0;

    int final_index = (ctrl->iterations == 0) ? ctrl->current_old : 1 - ctrl->current_old;
    double *u_final_sh = (final_index == 0) ? u_a : u_b;

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *u = (double *)malloc((size_t)n * sizeof(double));
    if (!x || !u) {
        free(x);
        free(u);
        for (int p = 0; p < num_processes; ++p) {
            sem_destroy(&start_sems[p]);
        }
        sem_destroy(&ctrl->done_sem);
        munmap(x_sh, (size_t)n * sizeof(double));
        munmap(u_a, (size_t)n * sizeof(double));
        munmap(u_b, (size_t)n * sizeof(double));
        munmap(ctrl, sizeof(SharedControl));
        munmap(start_sems, (size_t)num_processes * sizeof(sem_t));
        free(pids);
        return 0;
    }

    memcpy(x, x_sh, (size_t)n * sizeof(double));
    memcpy(u, u_final_sh, (size_t)n * sizeof(double));

    result->n = n;
    result->k = k;
    result->iterations = ctrl->iterations;
    result->converged = ctrl->converged;
    result->num_processes = num_processes;
    result->h = h;
    result->residual_rms = ctrl->residual_rms;
    result->error_rms = compute_error_rms(u, x, n);
    result->elapsed_seconds = elapsed;
    result->x = x;
    result->u = u;

    for (int p = 0; p < num_processes; ++p) {
        sem_destroy(&start_sems[p]);
    }
    sem_destroy(&ctrl->done_sem);
    munmap(x_sh, (size_t)n * sizeof(double));
    munmap(u_a, (size_t)n * sizeof(double));
    munmap(u_b, (size_t)n * sizeof(double));
    munmap(ctrl, sizeof(SharedControl));
    munmap(start_sems, (size_t)num_processes * sizeof(sem_t));
    free(pids);

    return 1;
}

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

static void print_summary_processes(const JacobiResult *res) {
    printf("===== Resultado Jacobi Poisson 1D (fork) =====\n");
    printf("k               : %d\n", res->k);
    printf("processes       : %d\n", res->num_processes);
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
           res->num_processes,
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
            "  %s solve_processes k tol max_iter num_processes\n"
            "  %s csv_serial k tol max_iter\n"
            "  %s csv_processes k tol max_iter num_processes\n",
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
            fprintf(stderr, "Error al ejecutar la version serial.\n");
            return 1;
        }

        print_summary_serial(&res);
        if (res.n <= 65) {
            print_solution_table(&res);
        }
        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "solve_processes") == 0) {
        if (argc != 6) {
            print_usage(argv[0]);
            return 1;
        }

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_processes(atoi(argv[2]), atof(argv[3]), atoi(argv[4]), atoi(argv[5]), &res)) {
            fprintf(stderr, "Error al ejecutar la version con procesos.\n");
            return 1;
        }

        print_summary_processes(&res);
        if (res.n <= 65) {
            print_solution_table(&res);
        }
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
        print_csv_line("serial", &res);
        free_result(&res);
        return 0;
    }

    if (strcmp(argv[1], "csv_processes") == 0) {
        if (argc != 6) {
            print_usage(argv[0]);
            return 1;
        }

        JacobiResult res = {0};
        if (!jacobi_poisson_1d_processes(atoi(argv[2]), atof(argv[3]), atoi(argv[4]), atoi(argv[5]), &res)) {
            return 1;
        }
        print_csv_line("processes", &res);
        free_result(&res);
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}