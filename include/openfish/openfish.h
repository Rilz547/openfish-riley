/** Riley Updates (Remove at the end)
 * @file openfish.h
 * @lastmodified: 2-slot pinned host decode ring (OPENFISH_HOST_RING) + host_slot on openfish_decode_gpu for P5-lite.
 * @lastpatched: 2026-07-18

******************************************************************************/

#ifndef OPENFISH_H
#define OPENFISH_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "openfish_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Depth of pinned host output ring. Slot 0 is enough for serial decode; overlap/P5-lite uses both. */
#define OPENFISH_HOST_RING 2

typedef struct openfish_gpubuf {
    float *bwd_NTC;
    float *post_NTC;
    uint8_t *moves;        /* device */
    char *sequence;        /* device */
    char *qstring;         /* device */
    void *beam_vector;
    void *states;
    float *qual_data;
    float *base_probs;
    float *total_probs;
    /* Persistent pinned host output ring (CUDA); NULL entries on HIP/Metal. */
    uint8_t *moves_host[OPENFISH_HOST_RING];
    char *sequence_host[OPENFISH_HOST_RING];
    char *qstring_host[OPENFISH_HOST_RING];
} openfish_gpubuf_t;

typedef struct openfish_opt {
    float beam_cut;
    float blank_score;
    float q_shift;
    float q_scale;
} openfish_opt_t;

// Emission-score element type of the input scores tensor.
//   OPENFISH_SCORE_F16 - native float scores (fp16 on the GPU path, fp32 on the CPU path);
//                        use score_scale = 1.0f for the unquantized pipeline.
//   OPENFISH_SCORE_I8  - int8 quantized scores in [-127, 127] (dorado-style); the raw value is
//                        dequantized on read as (float)s * score_scale (e.g. 5.0f/127.0f).
typedef enum {
    OPENFISH_SCORE_F16 = 0,
    OPENFISH_SCORE_I8  = 1
} openfish_score_dtype_t;

typedef struct openfish_decode_stats {
    double time_bwd_scan;
    double time_beam_search;
    double time_fwd_post_scan;
    double time_qual_data;
    double time_gen_sequence;
    double time_d2h_copy;
    int n_timesteps;
    int batch_size;
    int n_channels;
    /* Internal: CUDA event pairs for async (overlap) timing. Do not touch. */
    void *async_timing;
} openfish_decode_stats_t;

void openfish_decode_stats_reset(openfish_decode_stats_t *stats);

/* After the decode stream has been synchronized, resolve CUDA-event phase timers
 * recorded during an async (stream != NULL) openfish_decode_gpu call. */
void openfish_decode_stats_finish(openfish_decode_stats_t *stats);

openfish_opt_t openfish_decoder_default_opts(void);

void openfish_decode_cpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    int n_threads,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring,
    openfish_decode_stats_t *stats
);

void openfish_rotary_emb_cpu(
    void *x,
    const void *sin_buf,
    const void *cos_buf,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head,
    int n_threads
);

size_t openfish_gpubuf_size(
    int n_timesteps,
    int batch_size,
    int state_len
);

#if defined(HAVE_CUDA) || defined(HAVE_ROCM) || defined(HAVE_METAL)

void openfish_decode_gpu(
    int n_timesteps,
    int batch_size,
    int n_channels,
    const void *scores_NTC,
    openfish_score_dtype_t score_dtype,
    float score_scale,
    int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring,
    openfish_decode_stats_t *stats,
    void *stream,  /* cudaStream_t / hipStream_t; NULL = current/default stream */
    int host_slot  /* pinned host ring index in [0, OPENFISH_HOST_RING); serial path uses 0 */
);

/* Legacy hook after openfish_decode_gpu. On CUDA, host buffers are owned by gpubuf
 * (no-op). HIP/Metal may still free per-call malloc'd results. */
void openfish_decode_free_host(
    uint8_t *moves,
    char *sequence,
    char *qstring
);

openfish_gpubuf_t *openfish_gpubuf_init(
    int n_timesteps,
    int batch_size,
    int state_len
);

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
);

#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM) || defined(HAVE_METAL)

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

void openfish_rotary_emb_gpu(
    void *x,
    const void *sin_gpu,
    const void *cos_gpu,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head
);

void openfish_flstm_step_gpu(
    const void* scratch,
    const void* ih_t,
    void* cell,
    void* hh_next,
    int batch_size,
    int hidden_dim
);

void openfish_silu_mul_gpu(
    const void *in,
    void *out,
    int n_tokens,
    int hidden_dim
);

void openfish_rmsnorm_gpu(
    const void* in,
    const void* residual,
    const void* weight,
    void* out,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

void openfish_rmsnorm_quant_int8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

void openfish_rmsnorm_quant_fp8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

void openfish_quant_fp8_gpu(
    const void* in,
    void*       out,
    void*       scale,
    int         n_tokens,
    int         hidden_dim
);

void openfish_dequant_fp8_transpose_gpu(
    const void* in,
    void*       out,
    int         n_timesteps,
    int         batch_size,
    int         n_channels,
    float       scale
);

#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

#ifdef __cplusplus
}
#endif

#endif // OPENFISH_H
