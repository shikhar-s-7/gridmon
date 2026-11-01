/* gridmon C2000 port layer (M7), portable C99 part: MONITOR-ONLY.
 *
 * Control ISR at f_in (20/40/50/100 kHz)  --gm_port_push()-->  calibration trim (D-052) -> optional LN->LL
 *   -> polyphase anti-alias FIR + decimation to exactly 5 kHz -> int16 ring buffer (SPSC, 1.6 s)
 * Background loop / low-priority task / CPU2  --gm_port_background()-->  real-time trigger at 5 kHz
 *   -> record placement (estimated onset at ~0.28 s) -> gm_process_record (rules PRIMARY, GBM ADVISORY)
 *   -> telemetry struct (published, never written to any control variable).
 * The port never blocks the ISR and never writes outside its own struct. Simulation-only (Stage 1):
 * not validated on hardware (validation ladder L3+, docs/C2000_INTEGRATION.md).
 */
#ifndef GM_PORT_H
#define GM_PORT_H

#include <stdint.h>

#include "gm_port_fir.h"
#include "gridmon.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- sizes and scaling */
#define GM_PORT_RING_LEN 8000          /* 1.6 s at 5 kHz: 3 x 8000 int16 = 48,000 bytes (24,000 C28x words) */
#define GM_PORT_LSB (0x1p-14f)         /* ring/record scaling: int16 full scale = +/- 2 pu (finer than the 12-bit ADC) */
#define GM_PORT_INV_LSB (0x1p+14f)
#define GM_PORT_ONSET_TARGET 1400      /* place the estimated onset at sample 1400 = 0.28 s of the record */
#define GM_PORT_MAX_LOOKBACK 3599      /* onset estimates are clamped to <= 0.72 s before the trigger */
#define GM_PORT_HOLDOFF 5000           /* 1 s after a captured record before the trigger re-arms */
#define GM_PORT_WARMUP 5000            /* 1 s after init/resync before the trigger arms */

/* ---- trigger thresholds: deliberately BELOW the classification thresholds (fire early, classify later)
 * Urms(1/2) [0.92, 1.08] (rules 0.90/1.10); |J| 3 deg (rule 5 deg); |f_dft - f_ref| 0.15 Hz for 100 ms
 * (rule |df_end| 0.2 Hz); envelope peak deviation 1.5 % with sign alternation (rule: 1.0 % spectral peak,
 * see C2000_INTEGRATION.md: the class has depth >= 2 %). */
#define GM_PORT_TH_URMS_LO 0.92f
#define GM_PORT_TH_URMS_HI 1.08f
#define GM_PORT_TH_FREQ_HZ 0.15f
#define GM_PORT_FREQ_SUSTAIN 100       /* phasor samples (ms) */
#define GM_PORT_FREQ_TRACK_HZ 0.05f    /* f_ref (EMA, tau 1 s) is frozen while |f_dft - f_ref| >= this */
#define GM_PORT_FREQ_ALPHA 0.001f      /* EMA coefficient per ms */
#define GM_PORT_TH_ENV 0.015f          /* |e - mean_0.2s(e)| / e_ref */
#define GM_PORT_ENV_ALPHA 0.01f        /* envelope reference EMA per 10 ms (tau 1 s) */
#define GM_PORT_ENV_PAIR 1500          /* opposite-sign exceedances within 0.3 s (samples) */

/* trigger causes (bits) */
#define GM_TRIG_URMS 0x0001u
#define GM_TRIG_JUMP 0x0002u
#define GM_TRIG_FREQ 0x0004u
#define GM_TRIG_ENV  0x0008u

/* ---- configuration */
typedef struct {
    int32_t f_in_hz;     /* control-ISR rate: 20000, 40000, 50000 or 100000 */
    int16_t input_ln;    /* 0: inputs are v_ab, v_bc, v_ca (pu of nominal LL peak);
                            1 (GM_INPUT_LN): inputs are v_a, v_b, v_c (pu of nominal LN peak), v_ab = (v_a - v_b)/sqrt3 */
    float gain[3];       /* REQUIRED calibration trim per input channel (D-052): v = gain * (x - offset) */
    float offset[3];
} gm_port_cfg_t;

/* ---- telemetry (one per classified record) */
typedef struct {
    uint32_t seq;            /* published record counter (1, 2, ...) */
    uint32_t t_trigger;      /* 5 kHz sample counter at the trigger (timestamp; wraps after 9.9 days) */
    uint32_t t_record;       /* 5 kHz sample counter of record sample 0 */
    uint16_t cause;          /* GM_TRIG_* bits */
    int16_t status;          /* gm_process_record status */
    uint16_t flags;          /* gm_result_t flags */
    int16_t rule_label;      /* PRIMARY output (GM_CLASS_*) */
    int16_t rule_reason;
    float onset_s;           /* rules onset estimate (s from record start) */
    uint16_t onset_valid;
    float onset_est_s;       /* trigger's onset estimate (s from record start; target 0.28 s) */
    int16_t gbm_label;       /* ADVISORY output, -1 if GM_ENABLE_GBM = 0; never acted on */
    float gbm_max_proba;
    uint32_t overruns;       /* health: decimated samples dropped because the ring was full */
    uint32_t triggers;       /* health: trigger events (incl. those that started a capture) */
    uint32_t resyncs;        /* health: trigger resets after an overrun */
    uint32_t clipped;        /* health: records whose start had to be moved (history not available) */
} gm_telemetry_t;

/* ---- port state (caller-allocated; ~62 KB incl. the 48,000-byte ring; see gm_port_words()) */
typedef struct {
    /* ISR side: written only by gm_port_push */
    const float *fir_c;               /* [D][K] polyphase taps (gm_port_fir.c) */
    int16_t D, K, r, sb;              /* decimation, taps/phase, input phase, newest accumulator slot */
    int16_t input_ln;
    int16_t preroll;                  /* FIR pre-roll outputs still to discard (K - 1 at init) */
    uint16_t w;                       /* ring write position */
    float gain[3], offset[3];
    float acc[3][GM_PORT_K_MAX];      /* polyphase output accumulators */
    int16_t ring[GM_PORT_RING_LEN][3];
    volatile uint32_t head;           /* decimated samples written (5 kHz counter) */
    volatile uint32_t overruns;
    /* background side */
    volatile uint32_t tail;           /* oldest sample the background may still read */
    uint32_t cursor;                  /* next sample for the trigger */
    uint32_t seen_overruns;
    uint16_t rd;                      /* ring read position of cursor */
    int16_t state;                    /* GM_PORT_WARMUP_S, ARMED, CAPTURING, HOLDOFF */
    uint32_t state_until;             /* end of warm-up / hold-off */
    /* trigger: Urms(1/2) */
    float ur_acc[3], ur_prev[3];
    int16_t ur_cnt, ur_nblk;
    /* trigger: one-cycle DFT from 5-sample block sums (non-recursive, as DSP_SPEC 2) */
    float bl_re[3][20], bl_im[3][20], cu_re[3], cu_im[3];
    int16_t bl_i, bl_n, tw_i;
    uint32_t n_blocks;
    float ang[41];                    /* angle V1 history (ring) */
    float dfraw[20];                  /* df_raw history (ring) */
    float fdev[200];                  /* f_dft - 50 history (ring, 0.2 s) for back-extrapolation */
    int16_t ang_i, dfr_i, fdv_i;
    uint32_t n_phasors;
    float f_ref;
    int16_t f_ref_ok, freq_cnt;
    uint32_t freq_first;
    float freq_first_dev;
    /* trigger: envelope */
    float env_acc, env_hist[20], e_ref;
    int16_t env_n, env_i, env_cnt, env_sign;
    uint32_t env_last, env_first;
    /* capture */
    uint32_t rec_start, trig_n;
    uint16_t cause;
    float onset_est_s;
    uint32_t n_triggers, n_resyncs, n_clipped, seq;
    gm_ctx_t *ctx;                    /* caller's core context */
    int16_t *rec;                     /* caller's record buffer, 3 x 5000 int16 (planar) */
    gm_result_t res;
} gm_port_t;

enum { GM_PORT_WARMUP_S = 0, GM_PORT_ARMED = 1, GM_PORT_CAPTURING = 2, GM_PORT_HOLDOFF_S = 3 };

/* Initialise (also calls gm_init(ctx)). record_buf: 3 x 5000 int16. Returns GM_ERR_NULL / GM_ERR_LENGTH
 * (unsupported f_in) / GM_OK. */
gm_status_t gm_port_init(gm_port_t *p, const gm_port_cfg_t *cfg, gm_ctx_t *ctx, int16_t *record_buf);

/* ISR: one sample per channel at f_in (pu). ~3 + 3K MACs (K = 13) per call; never blocks. */
void gm_port_push(gm_port_t *p, float x0, float x1, float x2);

/* Background: consume decimated samples, trigger, capture, classify. Returns 1 and fills *tel when a
 * record was classified (call again: remaining samples are processed on the next call), else 0. */
int16_t gm_port_background(gm_port_t *p, gm_telemetry_t *tel);

/* Size of gm_port_t in 16-bit words. */
uint32_t gm_port_words(void);

#ifdef __cplusplus
}
#endif

#endif /* GM_PORT_H */
