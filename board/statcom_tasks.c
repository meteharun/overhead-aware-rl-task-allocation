/*
 * STATCOM task catalog. 25 tasks total.
 *
 * Tiers by period:
 *   t100_XX  100 us  HARD_RT   (11 tasks: t100_01..t100_11)
 *   t500_XX  500 us  FIRM_RT   ( 6 tasks: t500_01..t500_06)
 *   t1ms_XX  1000 us SOFT      ( 5 tasks: t1ms_01..t1ms_05)
 *   t2ms_XX  2000 us SOFT      ( 3 tasks: t2ms_01..t2ms_03)
 *
 * All: void t*_XX(float *in, float *out);
 *   in  points to at least 32 floats
 *   out points to at least 32 floats
 */

#define _GNU_SOURCE
#include <stdint.h>
#include <math.h>
#include <string.h>

#define BUFSZ 32

static inline void clarke(float a, float b, float c,
                          float *alpha, float *beta) {
    *alpha = a;
    *beta  = (b - c) * 0.57735f;
}

/* Reusable kernels */

static inline float dq_kernel(float *in, float *out, int N, float angle_step) {
    static float d_i = 0.0f, q_i = 0.0f;
    for (int i = 0; i < N; i++) {
        float a, b, d, q, s, cs;
        clarke(in[(i*3)%BUFSZ], in[(i*3+1)%BUFSZ], in[(i*3+2)%BUFSZ], &a, &b);
        sincosf(angle_step * i, &s, &cs);
        d =  a * cs + b * s;
        q = -a * s  + b * cs;
        d_i += (1.5f - d) * 1e-4f;
        q_i += (0.2f - q) * 1e-4f;
        out[i%BUFSZ] = d_i * 2.0f + q_i;
    }
    return d_i + q_i;
}

static inline float fir_kernel(float *in, int taps, int samples) {
    static float y[1024], h[128];
    static int initialized = 0;
    if (taps > 128) taps = 128;
    if (samples > 1024) samples = 1024;
    if (!initialized) {
        for (int i = 0; i < 128; i++) h[i] = 1.0f / (i + 1);
        initialized = 1;
    }
    for (int n = taps; n < samples; n++) {
        float acc = 0;
        for (int k = 0; k < taps; k++) acc += h[k] * in[(n-k) % BUFSZ];
        y[n] = acc;
    }
    return y[samples - 1];
}

static inline float matmul_kernel(float *in, int n) {
    static float A[96][96], B[96][96], C[96][96];
    if (n > 96) n = 96;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i][j] = in[(i+j) % BUFSZ] + 0.01f * i;
            B[i][j] = in[(i*3+j) % BUFSZ] * 0.7f;
        }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int k = 0; k < n; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
    float acc = 0;
    for (int i = 0; i < n; i++) acc += C[i][i];
    return acc;
}

/* ---- HARD_RT: 100 us ---- */

void t100_01(float *in, float *out) {
    out[0] = dq_kernel(in, out, 220, 0.003f);
}
void t100_02(float *in, float *out) {
    static float integ = 0.0f;
    float y = fir_kernel(in, 24, 200);
    float e = 400.0f - y;
    integ += e * 5e-4f;
    out[0] = e * 0.5f + integ;
}
void t100_03(float *in, float *out) {
    out[0] = matmul_kernel(in, 24);
}
void t100_04(float *in, float *out) {
    out[0] = matmul_kernel(in, 28);
}
void t100_05(float *in, float *out) {
    out[0] = dq_kernel(in, out, 250, 0.003f);
}
void t100_06(float *in, float *out) {
    static float integ = 0.0f;
    float y = fir_kernel(in, 20, 200);
    float e = 400.0f - y;
    integ += e * 5e-4f;
    out[0] = e * 0.5f + integ;
}
void t100_07(float *in, float *out) {
    out[0] = dq_kernel(in, out, 180, 0.0025f);
}
void t100_08(float *in, float *out) {
    out[0] = matmul_kernel(in, 20);
}
void t100_09(float *in, float *out) {
    static float integ = 0.0f;
    float y = fir_kernel(in, 32, 180);
    float e = 300.0f - y;
    integ += e * 4e-4f;
    out[0] = e * 0.6f + integ;
}
void t100_10(float *in, float *out) {
    out[0] = matmul_kernel(in, 26);
}
void t100_11(float *in, float *out) {
    out[0] = dq_kernel(in, out, 200, 0.002f);
}

/* ---- FIRM_RT: 500 us ---- */

void t500_01(float *in, float *out) {
    static float integ = 0.0f;
    float y = fir_kernel(in, 64, 512);
    float e = 400.0f - y;
    integ += e * 4e-4f;
    out[0] = e * 0.4f + integ;
}
void t500_02(float *in, float *out) {
    out[0] = matmul_kernel(in, 48);
}
void t500_03(float *in, float *out) {
    out[0] = matmul_kernel(in, 56);
}
void t500_04(float *in, float *out) {
    static float integ = 0.0f;
    float y = fir_kernel(in, 80, 768);
    float e = 400.0f - y;
    integ += e * 4e-4f;
    out[0] = e * 0.4f + integ;
}
void t500_05(float *in, float *out) {
    out[0] = matmul_kernel(in, 52);
}
void t500_06(float *in, float *out) {
    static float integ = 0.0f;
    float y = fir_kernel(in, 72, 640);
    float e = 350.0f - y;
    integ += e * 3e-4f;
    out[0] = e * 0.45f + integ;
}

/* ---- SOFT: 1 ms ---- */

void t1ms_01(float *in, float *out) {
    float y = fir_kernel(in, 128, 1024);
    static float d_i = 0.0f, q_i = 0.0f;
    d_i += (1.0f - y) * 1e-4f;
    q_i += (0.5f - y * 0.3f) * 1e-4f;
    out[0] = d_i;
    out[1] = q_i;
}
void t1ms_02(float *in, float *out) {
    out[0] = matmul_kernel(in, 64);
}
void t1ms_03(float *in, float *out) {
    static float integ = 0.0f;
    float y = fir_kernel(in, 100, 768);
    float e = 400.0f - y;
    integ += e * 3e-4f;
    out[0] = e * 0.3f + integ;
}
void t1ms_04(float *in, float *out) {
    out[0] = matmul_kernel(in, 72);
}
void t1ms_05(float *in, float *out) {
    float y = fir_kernel(in, 112, 896);
    out[0] = y + matmul_kernel(in, 40);
}

/* ---- SOFT: 2 ms ---- */

void t2ms_01(float *in, float *out) {
    float y1 = fir_kernel(in, 128, 1024);
    float y2 = fir_kernel(in, 96, 768);
    float m  = matmul_kernel(in, 48);
    out[0] = y1 + y2 + m;
}
void t2ms_02(float *in, float *out) {
    out[0] = matmul_kernel(in, 80);
}
void t2ms_03(float *in, float *out) {
    out[0] = matmul_kernel(in, 96);
}
