/* gm_cli: host tool around the portable core. Reads binary records (int16 little-endian/native,
 * planar [ab | bc | ca], 3 x 5000 samples per record) from a file or stdin and writes one JSON line
 * per record (rule label = PRIMARY output, GBM = ADVISORY). Simulation-only.
 *
 * Usage: gm_cli [--lsb PU_PER_COUNT] [--features] [FILE|-]
 *        gm_cli --version
 * Exit code: 0 ok, 1 usage/input error, 2 a record returned a non-OK status. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gm_names.h"
#include "gridmon.h"

static void print_float(FILE *f, float v)
{
    if (isfinite(v)) {
        fprintf(f, "%.9g", (double)v);
    } else {
        fputs("null", f);
    }
}

int main(int argc, char **argv)
{
    const char *path = "-";
    float lsb = 0x1p-10f;
    int with_features = 0, i, rc = 0;
    long rec = 0;
    FILE *in;
    gm_ctx_t *ctx;
    int16_t *x;
    gm_result_t r;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--version") == 0) {
            printf("%s\nsizeof(gm_ctx_t) = %lu bytes (%lu 16-bit words); sizeof(gm_result_t) = %lu bytes\n", gm_version(),
                   (unsigned long)sizeof(gm_ctx_t), (unsigned long)gm_ctx_words(), (unsigned long)sizeof(gm_result_t));
            return 0;
        } else if (strcmp(argv[i], "--lsb") == 0 && i + 1 < argc) {
            lsb = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--features") == 0) {
            with_features = 1;
        } else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            fprintf(stderr, "usage: gm_cli [--lsb PU] [--features] [FILE|-] | --version\n");
            return 1;
        } else {
            path = argv[i];
        }
    }
    in = strcmp(path, "-") == 0 ? stdin : fopen(path, "rb");
    if (in == NULL) {
        fprintf(stderr, "gm_cli: cannot open %s\n", path);
        return 1;
    }
    ctx = (gm_ctx_t *)malloc(sizeof(gm_ctx_t));
    x = (int16_t *)malloc(sizeof(int16_t) * GM_N_CH * GM_N_RECORD);
    if (ctx == NULL || x == NULL) {
        fprintf(stderr, "gm_cli: out of memory\n");
        return 1;
    }
    gm_init(ctx);
    while (fread(x, sizeof(int16_t), (size_t)(GM_N_CH * GM_N_RECORD), in) == (size_t)(GM_N_CH * GM_N_RECORD)) {
        gm_status_t st = gm_process_record(ctx, x, GM_N_RECORD, lsb, &r);
        if (st != GM_OK) {
            rc = 2;
        }
        printf("{\"record\":%ld,\"status\":%d,\"flags\":%u,\"rule_label\":\"%s\",\"rule_reason\":\"%s\",\"onset_s\":", rec,
               (int)r.status, (unsigned)r.flags, gm_class_names[r.rule_label], gm_reason_names[r.rule_reason]);
        if (r.onset_valid) {
            print_float(stdout, r.onset_estimate_s);
        } else {
            fputs("null", stdout);
        }
        if (r.gbm_label >= 0) {
            int k;
            printf(",\"gbm_label\":\"%s\",\"gbm_proba\":[", gm_class_names[r.gbm_label]);
            for (k = 0; k < GM_N_CLASSES; k++) {
                if (k) {
                    putchar(',');
                }
                print_float(stdout, r.gbm_proba[k]);
            }
            putchar(']');
        } else {
            fputs(",\"gbm_label\":null", stdout);
        }
        if (with_features) {
            int k;
            fputs(",\"features\":{", stdout);
            for (k = 0; k < GM_N_FEATURES; k++) {
                printf("%s\"%s\":", k ? "," : "", gm_feature_names[k]);
                if (r.feature_valid[k]) {
                    print_float(stdout, r.features[k]);
                } else {
                    fputs("null", stdout);
                }
            }
            fputs("},\"aux\":{", stdout);
            for (k = 0; k < GM_N_AUX; k++) {
                printf("%s\"%s\":", k ? "," : "", gm_aux_names[k]);
                if (r.aux_valid[k]) {
                    print_float(stdout, r.aux[k]);
                } else {
                    fputs("null", stdout);
                }
            }
            putchar('}');
        }
        fputs("}\n", stdout);
        rec++;
    }
    if (in != stdin) {
        fclose(in);
    }
    free(x);
    free(ctx);
    return rc;
}
