/* gridmon PIL template for LAUNCHXL-F28379D (validation ladder L3 preparation) -- UNTESTED.
 *
 * Runs gm_process_record on CPU1 for three embedded golden records (pil_records.c), measures the cycles of
 * each call with CPU Timer 1, prints label / reason / GBM label / cycles / max feature deviation from the
 * host C core (L1) over SCI-A (the LaunchPad's XDS100v2 virtual COM port, 115200 8N1) and a PASS/FAIL
 * line per record. Needs C2000Ware driverlib (device.h, driverlib.h). See README.md in this folder.
 * Simulation-only records; nothing is connected to any power stage.
 */
#include <stdint.h>

#include "device.h"
#include "driverlib.h"
#include "gridmon.h"
#include "pil_records.h"

/* large buffers in GS RAM (see the linker notes in README.md) */
#pragma DATA_SECTION(g_ctx, "ramgs_gridmon")
static gm_ctx_t g_ctx;
static gm_result_t g_res;

static void sci_init(void)
{
    GPIO_setMasterCore(DEVICE_GPIO_PIN_SCIRXDA, GPIO_CORE_CPU1);
    GPIO_setPinConfig(DEVICE_GPIO_CFG_SCIRXDA);
    GPIO_setDirectionMode(DEVICE_GPIO_PIN_SCIRXDA, GPIO_DIR_MODE_IN);
    GPIO_setPadConfig(DEVICE_GPIO_PIN_SCIRXDA, GPIO_PIN_TYPE_STD);
    GPIO_setQualificationMode(DEVICE_GPIO_PIN_SCIRXDA, GPIO_QUAL_ASYNC);
    GPIO_setMasterCore(DEVICE_GPIO_PIN_SCITXDA, GPIO_CORE_CPU1);
    GPIO_setPinConfig(DEVICE_GPIO_CFG_SCITXDA);
    GPIO_setDirectionMode(DEVICE_GPIO_PIN_SCITXDA, GPIO_DIR_MODE_OUT);
    GPIO_setPadConfig(DEVICE_GPIO_PIN_SCITXDA, GPIO_PIN_TYPE_STD);
    GPIO_setQualificationMode(DEVICE_GPIO_PIN_SCITXDA, GPIO_QUAL_ASYNC);
    SCI_performSoftwareReset(SCIA_BASE);
    SCI_setConfig(SCIA_BASE, DEVICE_LSPCLK_FREQ, 115200, (SCI_CONFIG_WLEN_8 | SCI_CONFIG_STOP_ONE | SCI_CONFIG_PAR_NONE));
    SCI_resetChannels(SCIA_BASE);
    SCI_enableFIFO(SCIA_BASE);
    SCI_enableModule(SCIA_BASE);
    SCI_performSoftwareReset(SCIA_BASE);
}

static void put_str(const char *s)
{
    while (*s != '\0') {
        SCI_writeCharBlockingFIFO(SCIA_BASE, (uint16_t)*s);
        s++;
    }
}

static void put_u32(uint32_t v)
{
    char buf[11];
    int16_t i = 10;
    buf[i] = '\0';
    do {
        buf[--i] = (char)('0' + (v % 10u));
        v /= 10u;
    } while (v != 0u && i > 0);
    put_str(&buf[i]);
}

static void put_i16(int16_t v)
{
    if (v < 0) {
        put_str("-");
        put_u32((uint32_t)(-(int32_t)v));
    } else {
        put_u32((uint32_t)v);
    }
}

void main(void)
{
    int16_t r, k, n_pass = 0;
    Device_init();
    Device_initGPIO();
    Interrupt_initModule();
    Interrupt_initVectorTable();
    sci_init();
    CPUTimer_setPeriod(CPUTIMER1_BASE, 0xFFFFFFFFu);
    CPUTimer_setPreScaler(CPUTIMER1_BASE, 0u);
    CPUTimer_stopTimer(CPUTIMER1_BASE);
    CPUTimer_reloadTimerCounter(CPUTIMER1_BASE);

    put_str("\r\ngridmon PIL (UNTESTED template): ");
    put_str(gm_version());
    put_str("\r\nctx words: ");
    put_u32(gm_ctx_words());
    put_str("\r\n");
    gm_init(&g_ctx);
    for (r = 0; r < PIL_N_RECORDS; r++) {
        uint32_t t0, t1;
        float dmax = 0.0f;
        int16_t ok;
        CPUTimer_reloadTimerCounter(CPUTIMER1_BASE);
        CPUTimer_startTimer(CPUTIMER1_BASE);
        t0 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        gm_process_record(&g_ctx, pil_x[r], GM_N_RECORD, PIL_LSB_PU, &g_res);
        t1 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        CPUTimer_stopTimer(CPUTIMER1_BASE);
        for (k = 0; k < GM_N_FEATURES; k++) {
            float d = g_res.features[k] - pil_expect_features[r][k];
            d = d < 0.0f ? -d : d;
            if (g_res.feature_valid[k] && d > dmax) {
                dmax = d;
            }
        }
        ok = (int16_t)(g_res.rule_label == pil_expect_rule_label[r] && g_res.rule_reason == pil_expect_rule_reason[r] &&
                       g_res.gbm_label == pil_expect_gbm_label[r]);
        n_pass = (int16_t)(n_pass + ok);
        put_str(pil_names[r]);
        put_str(": rule ");
        put_i16(g_res.rule_label);
        put_str(" reason ");
        put_i16(g_res.rule_reason);
        put_str(" gbm ");
        put_i16(g_res.gbm_label);
        put_str(" cycles ");
        put_u32(t0 - t1); /* down counter */
        put_str(" max|dfeature| x1e6 ");
        put_u32((uint32_t)(dmax * 1.0e6f));
        put_str(ok ? " PASS\r\n" : " FAIL\r\n");
    }
    put_str(n_pass == PIL_N_RECORDS ? "ALL PASS\r\n" : "SOME FAILED\r\n");
    for (;;) {
    }
}
