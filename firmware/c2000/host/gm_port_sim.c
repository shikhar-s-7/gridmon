/* Host simulation driver of the port layer (M7, tests/test_port_sil.py via ctypes). Not firmware.
 * Feeds input samples one at a time through gm_port_push (the "ISR") and calls gm_port_background every
 * bg_every input samples (the "background loop"); after a published record the background is blocked for
 * proc_delay input samples to emulate the target's record-processing time (overrun check). */
#include <stdlib.h>

#include "gm_port.h"

typedef struct {
    gm_port_t port;
    gm_ctx_t ctx;
    int16_t rec[3 * GM_N_RECORD];
    int32_t bg_every, bg_cnt, proc_delay, busy;
} gm_port_sim_t;

gm_port_sim_t *gm_port_sim_create(int32_t f_in, int16_t input_ln, const float *gain, const float *offset, int32_t bg_every,
                                  int32_t proc_delay)
{
    gm_port_cfg_t cfg;
    int32_t c;
    gm_port_sim_t *s = (gm_port_sim_t *)calloc(1, sizeof(gm_port_sim_t));
    if (s == NULL) {
        return NULL;
    }
    cfg.f_in_hz = f_in;
    cfg.input_ln = input_ln;
    for (c = 0; c < 3; c++) {
        cfg.gain[c] = gain[c];
        cfg.offset[c] = offset[c];
    }
    if (gm_port_init(&s->port, &cfg, &s->ctx, s->rec) != GM_OK) {
        free(s);
        return NULL;
    }
    s->bg_every = bg_every > 0 ? bg_every : 1;
    s->proc_delay = proc_delay;
    return s;
}

/* x: interleaved [n][3] pu. Returns the number of telemetry records written to tel (<= max_tel). */
int32_t gm_port_sim_feed(gm_port_sim_t *s, const float *x, int32_t n, gm_telemetry_t *tel, int32_t max_tel)
{
    int32_t i, k = 0;
    for (i = 0; i < n; i++) {
        gm_port_push(&s->port, x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        if (s->busy > 0) {
            s->busy--;
            continue;
        }
        if (++s->bg_cnt >= s->bg_every) {
            gm_telemetry_t t;
            s->bg_cnt = 0;
            while (gm_port_background(&s->port, &t)) {
                if (k < max_tel) {
                    tel[k++] = t;
                }
                s->busy = s->proc_delay;
                if (s->busy > 0) {
                    break;
                }
            }
        }
    }
    return k;
}

void gm_port_sim_destroy(gm_port_sim_t *s) { free(s); }
uint32_t gm_port_sim_sizeof_port(void) { return (uint32_t)sizeof(gm_port_t); }
uint32_t gm_port_sim_sizeof_telemetry(void) { return (uint32_t)sizeof(gm_telemetry_t); }
uint32_t gm_port_sim_port_words(void) { return gm_port_words(); }
uint32_t gm_port_sim_head(const gm_port_sim_t *s) { return s->port.head; }
uint32_t gm_port_sim_overruns(const gm_port_sim_t *s) { return s->port.overruns; }
uint32_t gm_port_sim_triggers(const gm_port_sim_t *s) { return s->port.n_triggers; }

/* Decimator check: push n inputs (interleaved [n][3]) and copy the decimated ring samples (int16 [m][3]). */
int32_t gm_port_sim_decimate(gm_port_sim_t *s, const float *x, int32_t n, int16_t *y, int32_t max_y)
{
    int32_t i, k = 0;
    for (i = 0; i < n; i++) {
        const uint32_t before = s->port.head;
        gm_port_push(&s->port, x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        if (s->port.head != before && k < max_y) {
            const uint16_t pos = (uint16_t)((s->port.w == 0u) ? (GM_PORT_RING_LEN - 1) : (s->port.w - 1u));
            y[3 * k] = s->port.ring[pos][0];
            y[3 * k + 1] = s->port.ring[pos][1];
            y[3 * k + 2] = s->port.ring[pos][2];
            k++;
        }
        s->port.tail = s->port.head; /* nobody reads: keep the ring from filling */
    }
    return k;
}
