/* gridmon C2000 adapter (M7): the thin, device-specific glue around the portable port layer (gm_port.c).
 * UNTESTED on hardware (validation ladder L3+). MONITOR-ONLY: nothing here may write a control variable,
 * a PWM register, a protection flag or any shared state of the control loop.
 *
 * Integration pattern (docs/C2000_INTEGRATION.md):
 *   1. In the existing control ISR (e.g. TIDA-01606 at 50 kHz), AFTER the control law has run and with the
 *      voltages it already converted to float pu, add exactly one call:
 *          gm_port_push(&gm_port, v_ab_pu, v_bc_pu, v_ca_pu);   (or v_a, v_b, v_c with cfg.input_ln = 1)
 *      Cost: ~3 + 3 x 13 float MACs (K = 13 at every supported f_in) plus a 3 x int16 store every D-th call.
 *   2. In the background loop (or a low-priority task, or on CPU2 of an F2837xD with the port struct in
 *      GSx RAM shared through the IPC/MSGRAM; then add a memory barrier before publishing head):
 *          if (gm_port_background(&gm_port, &tel)) { gm_c2000_publish(&tel); }
 *   3. Timestamps: the port counts decimated samples (5 kHz, wraps after 9.9 days). For wall-clock time,
 *      latch a free-running CPU timer (CPUTimer 1 or 2; CPU Timer 0 is often used by the SDK) at publish.
 * Memory (C28x words): gm_port_t ~31 K words (ring 24 K) -> GSx RAM; gm_ctx_t 7.9 K words; record buffer
 * 15 K words; result 174 words. Tables (DSP 2.7 K words, GBM 7 K words, FIR <= 2 K words) stay in flash
 * (or are copied to RAM for speed).
 */
#include "gm_port.h"

#if defined(__TMS320C28XX__) && defined(GM_C2000_DRIVERLIB)
#include "driverlib.h"
#include "device.h"
#define GM_TIMER_BASE CPUTIMER2_BASE
static uint32_t gm_c2000_now(void) { return 0xFFFFFFFFu - CPUTimer_getTimerCount(GM_TIMER_BASE); }
#else
static uint32_t gm_c2000_now(void) { return 0u; } /* host / no driverlib: no wall-clock time */
#endif

/* Telemetry sink: replace with SCI/CAN/IPC transmission in the application. It receives a copy; it must not
 * block the background loop for long (queue it). The default keeps nothing (no static mutable state). */
void gm_c2000_publish(const gm_telemetry_t *tel, uint32_t *timestamp_out);

void gm_c2000_publish(const gm_telemetry_t *tel, uint32_t *timestamp_out)
{
    (void)tel;
    if (timestamp_out != 0) {
        *timestamp_out = gm_c2000_now();
    }
}

/* Example configuration for TIDA-01606 (three-phase, 50 kHz control ISR, LN sensing) -- values are
 * placeholders until the per-channel calibration (D-052, C2000_INTEGRATION.md section 4) is done. */
void gm_c2000_default_cfg(gm_port_cfg_t *cfg);

void gm_c2000_default_cfg(gm_port_cfg_t *cfg)
{
    int16_t c;
    cfg->f_in_hz = 50000;
    cfg->input_ln = 1;
    for (c = 0; c < 3; c++) {
        cfg->gain[c] = 1.0f;   /* REPLACE with the measured trim */
        cfg->offset[c] = 0.0f; /* REPLACE with the measured offset (pu) */
    }
}
