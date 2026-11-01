/* gridmon C2000 port layer (M7), portable C99 part. See gm_port.h and docs/C2000_INTEGRATION.md.
 *
 * ISR path (gm_port_push, every input sample at f_in):
 *   x_c <- gain_c (x_c - offset_c)                         calibration trim (D-052), per input channel
 *   LN mode: v_ab = (x_a - x_b)/sqrt3, v_bc = (x_b - x_c)/sqrt3, v_ca = (x_c - x_a)/sqrt3
 *   polyphase FIR: y[k] = sum_i h[i] v[D k + L - 1 - i]; K accumulators, K MACs per channel per input
 *   every D inputs (after K - 1 discarded pre-roll outputs, so ring sample n = FIR output k = n): y -> int16 (LSB 2^-14 pu, round half away from zero, saturate) -> ring[head]; head++
 *   ring full (head - tail >= 8000): sample dropped, overruns++ (the ISR never waits).
 * Background (gm_port_background):
 *   trigger at 5 kHz: Urms(1/2) outside [0.92, 1.08] (window 100, hop 50); one-cycle DFT phasors at
 *   1 kHz from 5-sample block sums -> V1 -> |J| > 3 deg; f_dft (DSP_SPEC 3) vs f_ref (EMA tau 1 s, frozen
 *   while |dev| >= 0.05 Hz, re-armed after each hold-off) beyond 0.15 Hz for 100 ms; envelope (10-phasor
 *   means of |V1|) minus its 0.2 s mean beyond 1.5 % of e_ref with opposite signs within 0.3 s.
 *   Onset estimate: Urms -> window centre; J -> trigger - 10 ms; frequency -> first exceedance
 *   - 0.15 Hz / ROCOF_est - 30 ms filter delay (ROCOF_est = max |df/dt| over the last 0.2 s);
 *   envelope -> first exceedance - 25 ms. Record start = onset_est - 0.28 s (history <= 1 s: ring 1.6 s).
 *   When the record end is in the ring: copy (planar) -> gm_process_record -> telemetry; hold-off 1 s.
 */
#include "gm_port.h"

#include <limits.h>

#include "gm_internal.h"

static uint16_t gm_ring_next(uint16_t i)
{
    return (uint16_t)((i + 1u == (uint16_t)GM_PORT_RING_LEN) ? 0u : (i + 1u));
}

uint32_t gm_port_words(void)
{
    return (uint32_t)((sizeof(gm_port_t) * (uint32_t)CHAR_BIT) / 16u);
}

static void gm_trigger_reset(gm_port_t *p)
{
    int32_t c, i;
    for (c = 0; c < 3; c++) {
        p->ur_acc[c] = 0.0f;
        p->ur_prev[c] = 0.0f;
        p->cu_re[c] = 0.0f;
        p->cu_im[c] = 0.0f;
        for (i = 0; i < 20; i++) {
            p->bl_re[c][i] = 0.0f;
            p->bl_im[c][i] = 0.0f;
        }
    }
    for (i = 0; i < 41; i++) {
        p->ang[i] = 0.0f;
    }
    for (i = 0; i < 20; i++) {
        p->dfraw[i] = 0.0f;
        p->env_hist[i] = 0.0f;
    }
    for (i = 0; i < 200; i++) {
        p->fdev[i] = 0.0f;
    }
    p->ur_cnt = 0;
    p->ur_nblk = 0;
    p->bl_i = 0;
    p->bl_n = 0;
    p->tw_i = (int16_t)(p->cursor % (uint32_t)GM_N_CYCLE);
    p->n_blocks = 0u;
    p->ang_i = 0;
    p->dfr_i = 0;
    p->fdv_i = 0;
    p->n_phasors = 0u;
    p->f_ref = 0.0f;
    p->f_ref_ok = 0;
    p->freq_cnt = 0;
    p->freq_first = 0u;
    p->freq_first_dev = 0.0f;
    p->env_acc = 0.0f;
    p->e_ref = 0.0f;
    p->env_n = 0;
    p->env_i = 0;
    p->env_cnt = 0;
    p->env_sign = 0;
    p->env_last = 0u;
    p->env_first = 0u;
    p->state = GM_PORT_WARMUP_S;
    p->state_until = p->cursor + (uint32_t)GM_PORT_WARMUP;
}

gm_status_t gm_port_init(gm_port_t *p, const gm_port_cfg_t *cfg, gm_ctx_t *ctx, int16_t *record_buf)
{
    int32_t i, c;
    if (p == 0 || cfg == 0 || ctx == 0 || record_buf == 0) {
        return GM_ERR_NULL;
    }
    p->fir_c = 0;
    for (i = 0; i < GM_PORT_N_FIR; i++) {
        if (gm_port_firs[i].f_in_hz == cfg->f_in_hz) {
            p->fir_c = gm_port_firs[i].c;
            p->D = gm_port_firs[i].D;
            p->K = gm_port_firs[i].K;
            p->r = (int16_t)(gm_port_firs[i].pad % gm_port_firs[i].D); /* input 0 has phase pad mod D (pad < D) */
        }
    }
    if (p->fir_c == 0) {
        return GM_ERR_LENGTH;
    }
    p->sb = 0;
    p->preroll = (int16_t)(p->K - 1);
    p->input_ln = cfg->input_ln ? 1 : 0;
    p->w = 0u;
    for (c = 0; c < 3; c++) {
        p->gain[c] = cfg->gain[c];
        p->offset[c] = cfg->offset[c];
        for (i = 0; i < GM_PORT_K_MAX; i++) {
            p->acc[c][i] = 0.0f;
        }
    }
    for (i = 0; i < GM_PORT_RING_LEN; i++) {
        p->ring[i][0] = 0;
        p->ring[i][1] = 0;
        p->ring[i][2] = 0;
    }
    p->head = 0u;
    p->overruns = 0u;
    p->tail = 0u;
    p->cursor = 0u;
    p->seen_overruns = 0u;
    p->rd = 0u;
    p->rec_start = 0u;
    p->trig_n = 0u;
    p->cause = 0u;
    p->onset_est_s = 0.0f;
    p->n_triggers = 0u;
    p->n_resyncs = 0u;
    p->n_clipped = 0u;
    p->seq = 0u;
    p->ctx = ctx;
    p->rec = record_buf;
    gm_trigger_reset(p);
    return gm_init(ctx);
}

/* ------------------------------------------------------------------------------------------- ISR side */
static int16_t gm_to_count(float y)
{
    float q = y * GM_PORT_INV_LSB;
    q += (q >= 0.0f) ? 0.5f : -0.5f;
    if (q > 32767.0f) {
        q = 32767.0f;
    } else if (q < -32767.0f) {
        q = -32767.0f;
    }
    return (int16_t)(int32_t)q;
}

void gm_port_push(gm_port_t *p, float x0, float x1, float x2)
{
    float v0, v1, v2;
    const float *c;
    int16_t m, slot;
    x0 = (x0 - p->offset[0]) * p->gain[0];
    x1 = (x1 - p->offset[1]) * p->gain[1];
    x2 = (x2 - p->offset[2]) * p->gain[2];
    if (p->input_ln) {
        v0 = (x0 - x1) * GM_K_INV_SQRT3;
        v1 = (x1 - x2) * GM_K_INV_SQRT3;
        v2 = (x2 - x0) * GM_K_INV_SQRT3;
    } else {
        v0 = x0;
        v1 = x1;
        v2 = x2;
    }
    c = p->fir_c + (int32_t)p->r * (int32_t)p->K;
    slot = p->sb;
    for (m = 0; m < p->K; m++) { /* output (newest - m) lives in slot (sb - m) mod K */
        p->acc[0][slot] += c[m] * v0;
        p->acc[1][slot] += c[m] * v1;
        p->acc[2][slot] += c[m] * v2;
        slot = (int16_t)((slot == 0) ? (p->K - 1) : (slot - 1));
    }
    if (p->r == p->D - 1) { /* oldest output complete: slot (sb + 1) mod K */
        const int16_t e = (int16_t)((p->sb + 1 == p->K) ? 0 : (p->sb + 1));
        const uint32_t h = p->head;
        if (p->preroll > 0) { /* outputs k < 0 (window before the first input): discard */
            p->preroll--;
        } else if ((uint32_t)(h - p->tail) < (uint32_t)GM_PORT_RING_LEN) {
            p->ring[p->w][0] = gm_to_count(p->acc[0][e]);
            p->ring[p->w][1] = gm_to_count(p->acc[1][e]);
            p->ring[p->w][2] = gm_to_count(p->acc[2][e]);
            p->w = gm_ring_next(p->w);
            p->head = h + 1u; /* publish after the data (single writer) */
        } else {
            p->overruns = p->overruns + 1u;
        }
        p->acc[0][e] = 0.0f;
        p->acc[1][e] = 0.0f;
        p->acc[2][e] = 0.0f;
        p->sb = e;
        p->r = 0;
    } else {
        p->r++;
    }
}

/* ------------------------------------------------------------------------------------------- background */
static uint16_t gm_trig_urms(gm_port_t *p, const float v[3], uint32_t n, uint32_t *onset)
{
    int32_t c;
    uint16_t fire = 0u;
    for (c = 0; c < 3; c++) {
        p->ur_acc[c] += v[c] * v[c];
    }
    if (++p->ur_cnt < GM_URMS_HOP) {
        return 0u;
    }
    p->ur_cnt = 0;
    for (c = 0; c < 3; c++) {
        if (p->ur_nblk > 0) {
            const float u = GM_SQRTF(GM_K_URMS_SCALE * (p->ur_prev[c] + p->ur_acc[c]));
            if (u < GM_PORT_TH_URMS_LO || u > GM_PORT_TH_URMS_HI) {
                fire = 1u;
            }
        }
        p->ur_prev[c] = p->ur_acc[c];
        p->ur_acc[c] = 0.0f;
    }
    if (p->ur_nblk < 2) {
        p->ur_nblk++;
    }
    if (fire) {
        *onset = n - 60u; /* within the 100-sample window, biased to its second half */
        return GM_TRIG_URMS;
    }
    return 0u;
}

static float gm_ring_get(const float *buf, int16_t newest, int16_t len, int16_t back)
{
    int16_t i = (int16_t)(newest - back);
    if (i < 0) {
        i = (int16_t)(i + len);
    }
    return buf[i];
}

/* 1 kHz phasor update: J, f_dft, envelope */
static uint16_t gm_trig_phasor(gm_port_t *p, float a1, float ang, uint32_t n, uint32_t *onset)
{
    uint16_t fire = 0u;
    int32_t j;
    p->ang_i = (int16_t)((p->ang_i + 1 == 41) ? 0 : (p->ang_i + 1));
    p->ang[p->ang_i] = ang;
    p->n_phasors++;
    if (p->n_phasors > 40u) {
        const float a20 = gm_ring_get(p->ang, p->ang_i, 41, 20);
        const float a40 = gm_ring_get(p->ang, p->ang_i, 41, 40);
        const float J = gm_wrap((ang - 2.0f * a20) + a40);
        if (GM_FABSF(J) > GM_TH_JUMP_ONSET_RAD_GT) {
            fire |= GM_TRIG_JUMP;
            *onset = n - 50u;
        }
    }
    if (p->n_phasors > 20u) {
        const float a20 = gm_ring_get(p->ang, p->ang_i, 41, 20);
        p->dfr_i = (int16_t)((p->dfr_i + 1 == 20) ? 0 : (p->dfr_i + 1));
        p->dfraw[p->dfr_i] = gm_wrap(ang - a20) * GM_K_DF_SCALE;
    }
    if (p->n_phasors > 40u) {
        float s = 0.0f, dev, rmax = 0.0f;
        for (j = 0; j < 20; j++) {
            s += p->dfraw[j];
        }
        s *= GM_K_MA_INV; /* f_dft - 50 */
        p->fdv_i = (int16_t)((p->fdv_i + 1 == 200) ? 0 : (p->fdv_i + 1));
        p->fdev[p->fdv_i] = s;
        if (!p->f_ref_ok) {
            p->f_ref = s;
            p->f_ref_ok = 1;
        }
        dev = s - p->f_ref;
        if (GM_FABSF(dev) < GM_PORT_FREQ_TRACK_HZ) {
            p->f_ref += GM_PORT_FREQ_ALPHA * dev;
        }
        if (GM_FABSF(dev) > GM_PORT_TH_FREQ_HZ) {
            if (p->freq_cnt == 0) {
                p->freq_first = n;
                p->freq_first_dev = GM_FABSF(dev);
            }
            if (p->freq_cnt < GM_PORT_FREQ_SUSTAIN) {
                p->freq_cnt++;
            }
            if (p->freq_cnt >= GM_PORT_FREQ_SUSTAIN && p->n_phasors > 240u) {
                uint32_t back;
                for (j = 0; j < 180; j++) { /* ROCOF_est = max |f(m) - f(m-20)| / 0.02 s over 0.2 s */
                    const float d = GM_FABSF(gm_ring_get(p->fdev, p->fdv_i, 200, (int16_t)j) -
                                             gm_ring_get(p->fdev, p->fdv_i, 200, (int16_t)(j + 20)));
                    if (d > rmax) {
                        rmax = d;
                    }
                }
                rmax *= 50.0f; /* Hz/s */
                if (rmax < 0.2f) {
                    rmax = 0.2f;
                }
                back = (uint32_t)(int32_t)(GM_DIVF(p->freq_first_dev, rmax) * 5000.0f) + 150u;
                *onset = p->freq_first - back;
                fire |= GM_TRIG_FREQ;
            }
        } else {
            p->freq_cnt = 0;
        }
    }
    /* envelope: 10-phasor means of |V1| at 100 Hz */
    p->env_acc += a1;
    if (++p->env_n == GM_ENV_BLOCK) {
        const float e = p->env_acc * GM_K_BLOCK_INV;
        p->env_acc = 0.0f;
        p->env_n = 0;
        if (p->env_cnt >= 20) {
            float mean = 0.0f, d;
            for (j = 0; j < 20; j++) {
                mean += p->env_hist[j];
            }
            mean *= 0.05f;
            d = e - mean;
            if (p->e_ref <= 0.0f) {
                p->e_ref = mean;
            }
            if (GM_FABSF(d) < GM_K_HALF * GM_PORT_TH_ENV * p->e_ref) {
                p->e_ref += GM_PORT_ENV_ALPHA * (e - p->e_ref);
            }
            if (GM_FABSF(d) > GM_PORT_TH_ENV * p->e_ref) {
                const int16_t sg = (int16_t)(d > 0.0f ? 1 : -1);
                if (p->env_sign != 0 && sg != p->env_sign && (n - p->env_last) <= (uint32_t)GM_PORT_ENV_PAIR) {
                    fire |= GM_TRIG_ENV;
                    if (!(fire & (GM_TRIG_JUMP | GM_TRIG_FREQ))) {
                        *onset = p->env_first - 125u;
                    }
                }
                if (p->env_sign == 0 || (n - p->env_last) > (uint32_t)GM_PORT_ENV_PAIR) {
                    p->env_first = n;
                }
                p->env_sign = sg;
                p->env_last = n;
            }
        } else {
            p->env_cnt++;
        }
        p->env_i = (int16_t)((p->env_i + 1 == 20) ? 0 : (p->env_i + 1));
        p->env_hist[p->env_i] = e;
    }
    return fire;
}

/* one 5 kHz sample through the trigger; returns cause bits, *onset = estimated onset sample */
static uint16_t gm_trigger(gm_port_t *p, const float v[3], uint32_t n, uint32_t *onset)
{
    uint16_t cause;
    uint32_t on_u = n, on_p = n;
    int32_t c, b;
    cause = gm_trig_urms(p, v, n, &on_u);
    for (c = 0; c < 3; c++) {
        p->cu_re[c] += v[c] * gm_tw100_re[p->tw_i];
        p->cu_im[c] += v[c] * gm_tw100_im[p->tw_i];
    }
    p->tw_i = (int16_t)((p->tw_i + 1 == GM_N_CYCLE) ? 0 : (p->tw_i + 1));
    if (++p->bl_n == GM_STEP) {
        p->bl_n = 0;
        for (c = 0; c < 3; c++) {
            p->bl_re[c][p->bl_i] = p->cu_re[c];
            p->bl_im[c][p->bl_i] = p->cu_im[c];
            p->cu_re[c] = 0.0f;
            p->cu_im[c] = 0.0f;
        }
        p->bl_i = (int16_t)((p->bl_i + 1 == 20) ? 0 : (p->bl_i + 1));
        p->n_blocks++;
        if (p->n_blocks >= 20u) {
            float xr[3], xi[3], v1r, v1i;
            for (c = 0; c < 3; c++) {
                float sr = 0.0f, si = 0.0f;
                for (b = 0; b < 20; b++) {
                    sr += p->bl_re[c][b];
                    si += p->bl_im[c][b];
                }
                xr[c] = GM_K_DFT_SCALE * sr;
                xi[c] = GM_K_DFT_SCALE * si;
            }
            v1r = (xr[0] + (GM_K_A_RE * xr[1] - GM_K_A_IM * xi[1]) + (GM_K_A2_RE * xr[2] - GM_K_A2_IM * xi[2])) * GM_K_THIRD;
            v1i = (xi[0] + (GM_K_A_RE * xi[1] + GM_K_A_IM * xr[1]) + (GM_K_A2_RE * xi[2] + GM_K_A2_IM * xr[2])) * GM_K_THIRD;
            cause |= gm_trig_phasor(p, GM_SQRTF(v1r * v1r + v1i * v1i), GM_ATAN2F(v1i, v1r), n, &on_p);
        }
    }
    if (cause & GM_TRIG_URMS) {
        *onset = on_u;
    }
    if ((cause & ~GM_TRIG_URMS) && (!(cause & GM_TRIG_URMS) || on_p < on_u)) {
        *onset = on_p;
    }
    return cause;
}

static void gm_capture(gm_port_t *p, gm_telemetry_t *tel)
{
    int32_t i, c;
    uint16_t pos = (uint16_t)(p->rd); /* rd = position of cursor (the record's last sample + 1) */
    float mp = 0.0f;
    /* record occupies [rec_start, rec_start + 5000) = the 5000 samples ending at cursor - 1 */
    for (i = 0; i < GM_N_RECORD; i++) {
        pos = (uint16_t)((pos == 0u) ? (GM_PORT_RING_LEN - 1) : (pos - 1u));
        for (c = 0; c < 3; c++) {
            p->rec[c * GM_N_RECORD + (GM_N_RECORD - 1 - i)] = p->ring[pos][c];
        }
    }
    gm_process_record(p->ctx, p->rec, GM_N_RECORD, GM_PORT_LSB, &p->res);
    p->seq++;
    tel->seq = p->seq;
    tel->t_trigger = p->trig_n;
    tel->t_record = p->rec_start;
    tel->cause = p->cause;
    tel->status = p->res.status;
    tel->flags = p->res.flags;
    tel->rule_label = p->res.rule_label;
    tel->rule_reason = p->res.rule_reason;
    tel->onset_s = p->res.onset_estimate_s;
    tel->onset_valid = p->res.onset_valid;
    tel->onset_est_s = p->onset_est_s;
    tel->gbm_label = p->res.gbm_label;
    for (i = 0; i < GM_N_CLASSES; i++) {
        if (p->res.gbm_proba[i] > mp) {
            mp = p->res.gbm_proba[i];
        }
    }
    tel->gbm_max_proba = mp;
    tel->overruns = p->overruns;
    tel->triggers = p->n_triggers;
    tel->resyncs = p->n_resyncs;
    tel->clipped = p->n_clipped;
}

int16_t gm_port_background(gm_port_t *p, gm_telemetry_t *tel)
{
    const uint32_t head = p->head;
    if (p->overruns != p->seen_overruns) { /* gap in the decimated stream: restart the trigger */
        p->seen_overruns = p->overruns;
        p->n_resyncs++;
        gm_trigger_reset(p);
    }
    while (p->cursor != head) {
        const uint32_t n = p->cursor;
        float v[3];
        uint32_t onset = n;
        uint16_t cause;
        v[0] = GM_PORT_LSB * (float)p->ring[p->rd][0];
        v[1] = GM_PORT_LSB * (float)p->ring[p->rd][1];
        v[2] = GM_PORT_LSB * (float)p->ring[p->rd][2];
        p->rd = gm_ring_next(p->rd);
        p->cursor = n + 1u;
        cause = gm_trigger(p, v, n, &onset);
        if (p->state == GM_PORT_WARMUP_S || p->state == GM_PORT_HOLDOFF_S) {
            if (n + 1u >= p->state_until) { /* (re-)arm with fresh references */
                p->state = GM_PORT_ARMED;
                p->f_ref_ok = 0;
                p->freq_cnt = 0;
                p->env_sign = 0;
                p->e_ref = 0.0f;
            }
        } else if (p->state == GM_PORT_ARMED && cause != 0u) {
            uint32_t oldest = (head > (uint32_t)GM_PORT_RING_LEN - 16u) ? head - ((uint32_t)GM_PORT_RING_LEN - 16u) : 0u;
            p->n_triggers++;
            if ((uint32_t)(n - onset) > (uint32_t)GM_PORT_MAX_LOOKBACK) {
                onset = n - (uint32_t)GM_PORT_MAX_LOOKBACK;
            }
            p->rec_start = (onset >= (uint32_t)GM_PORT_ONSET_TARGET) ? onset - (uint32_t)GM_PORT_ONSET_TARGET : 0u;
            if (p->rec_start < oldest) {
                p->rec_start = oldest;
                p->n_clipped++;
            }
            p->onset_est_s = GM_DIVF((float)(int32_t)(onset - p->rec_start), GM_K_FS);
            p->trig_n = n;
            p->cause = cause;
            p->state = GM_PORT_CAPTURING;
        }
        if (p->state == GM_PORT_CAPTURING && p->cursor >= p->rec_start + (uint32_t)GM_N_RECORD) {
            gm_capture(p, tel);
            p->state = GM_PORT_HOLDOFF_S;
            p->state_until = p->cursor + (uint32_t)GM_PORT_HOLDOFF;
            p->tail = (p->cursor > 5000u) ? p->cursor - 5000u : 0u;
            return 1;
        }
        if (p->state == GM_PORT_CAPTURING) {
            p->tail = p->rec_start;
        } else {
            p->tail = (p->cursor > 5000u) ? p->cursor - 5000u : 0u;
        }
    }
    return 0;
}
