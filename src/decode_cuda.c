/** Riley Updates (Remove at the end)
 * @file decode_cuda.c
 * @lastmodified: 2-slot pinned host ring; openfish_decode_gpu takes host_slot for P5-lite overlap.
 * @lastpatched: 2026-07-18

******************************************************************************/

#include <openfish/openfish.h>
#include <stdint.h>
#include "openfish_defs.h"
#include "scan_cuda.h"
#include "beam_search_cuda.h"
#include "error.h"
#include "cuda_utils.h"
#include "misc.h"

#include <openfish/openfish_error.h>

#include <cuda_fp16.h>

#define OPENFISH_DECODE_MAX_PHASES 8

typedef struct {
    cudaEvent_t start[OPENFISH_DECODE_MAX_PHASES];
    cudaEvent_t end[OPENFISH_DECODE_MAX_PHASES];
    double *accum[OPENFISH_DECODE_MAX_PHASES];
    int n;
} openfish_cuda_async_timing_t;

static void openfish_decode_stats_note_dims(
    openfish_decode_stats_t *stats,
    int n_timesteps,
    int batch_size,
    int n_channels
) {
    if (stats == NULL) {
        return;
    }
    stats->n_timesteps = n_timesteps;
    stats->batch_size = batch_size;
    stats->n_channels = n_channels;
}

static openfish_cuda_async_timing_t *openfish_async_timing_ensure(openfish_decode_stats_t *stats) {
    if (stats->async_timing == NULL) {
        openfish_cuda_async_timing_t *t =
            (openfish_cuda_async_timing_t *)calloc(1, sizeof(openfish_cuda_async_timing_t));
        MALLOC_CHK(t);
        stats->async_timing = t;
    }
    return (openfish_cuda_async_timing_t *)stats->async_timing;
}

static void openfish_async_timing_record(
    openfish_decode_stats_t *stats,
    double *accum_field,
    cudaStream_t stream,
    int is_start
) {
    openfish_cuda_async_timing_t *t = openfish_async_timing_ensure(stats);
    if (is_start) {
        if (t->n >= OPENFISH_DECODE_MAX_PHASES) {
            OPENFISH_ERROR("%s", "too many decode timing phases");
            exit(EXIT_FAILURE);
        }
        cudaEventCreate(&t->start[t->n]);
        checkCudaError();
        cudaEventCreate(&t->end[t->n]);
        checkCudaError();
        t->accum[t->n] = accum_field;
        cudaEventRecord(t->start[t->n], stream);
        checkCudaError();
    } else {
        cudaEventRecord(t->end[t->n], stream);
        checkCudaError();
        t->n += 1;
    }
}

/* Host-wall timing (serial path). */
#define OPENFISH_TIME_SECTION(stats_ptr, field, ...) \
    do { \
        double _openfish_t0 = 0.0; \
        double _openfish_t1 = 0.0; \
        if ((stats_ptr) != NULL) { \
            _openfish_t0 = openfish_realtime(); \
        } \
        __VA_ARGS__ \
        if ((stats_ptr) != NULL) { \
            _openfish_t1 = openfish_realtime(); \
            (stats_ptr)->field += _openfish_t1 - _openfish_t0; \
        } \
    } while (0)

/* CUDA-event timing for overlap path: record only; resolve in openfish_decode_stats_finish. */
#define OPENFISH_TIME_SECTION_ASYNC(stats_ptr, field, stream_handle, ...) \
    do { \
        if ((stats_ptr) != NULL) { \
            openfish_async_timing_record((stats_ptr), &(stats_ptr)->field, (stream_handle), 1); \
        } \
        __VA_ARGS__ \
        if ((stats_ptr) != NULL) { \
            openfish_async_timing_record((stats_ptr), &(stats_ptr)->field, (stream_handle), 0); \
        } \
    } while (0)

void openfish_decode_stats_finish(openfish_decode_stats_t *stats) {
    if (stats == NULL || stats->async_timing == NULL) {
        return;
    }
    openfish_cuda_async_timing_t *t = (openfish_cuda_async_timing_t *)stats->async_timing;
    for (int i = 0; i < t->n; ++i) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t->start[i], t->end[i]);
        checkCudaError();
        if (t->accum[i] != NULL) {
            *(t->accum[i]) += (double)ms / 1000.0;
        }
        cudaEventDestroy(t->start[i]);
        cudaEventDestroy(t->end[i]);
    }
    free(t);
    stats->async_timing = NULL;
}

openfish_gpubuf_t *openfish_gpubuf_init(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(malloc(sizeof(openfish_gpubuf_t)));
    MALLOC_CHK(gpubuf);

    const int num_states = pow(NUM_BASES, state_len);

    // scan tensors
    cudaMalloc((void **)&gpubuf->bwd_NTC, sizeof(float) * batch_size * (n_timesteps + 1) * num_states);
	checkCudaError();
    cudaMalloc((void **)&gpubuf->post_NTC, sizeof(float) * batch_size * (n_timesteps + 1) * num_states);
	checkCudaError();

    // return buffers
    cudaMalloc((void **)&gpubuf->moves, sizeof(uint8_t) * batch_size * n_timesteps);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->sequence, sizeof(char) * batch_size * n_timesteps);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->qstring, sizeof(char) * batch_size * n_timesteps);
    checkCudaError();

    // beamsearch buffers
    cudaMalloc((void **)&gpubuf->beam_vector, sizeof(beam_element_t) * batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1));
    checkCudaError();
    cudaMalloc((void **)&gpubuf->states, sizeof(state_t) * batch_size * n_timesteps);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->qual_data, sizeof(float) * batch_size * n_timesteps * NUM_BASES);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->base_probs, sizeof(float) * batch_size * n_timesteps);
    checkCudaError();
    cudaMalloc((void **)&gpubuf->total_probs, sizeof(float) * batch_size * n_timesteps);
    checkCudaError();

    // Persistent pinned host return ring (2 slots for P5-lite; frees in openfish_gpubuf_free).
    cudaError_t herr;
    for (int slot = 0; slot < OPENFISH_HOST_RING; ++slot) {
        herr = cudaHostAlloc((void **)&gpubuf->moves_host[slot], sizeof(uint8_t) * batch_size * n_timesteps, cudaHostAllocDefault);
        if (herr != cudaSuccess) {
            OPENFISH_ERROR("cudaHostAlloc(moves_host[%d]) failed: %s", slot, cudaGetErrorString(herr));
            exit(EXIT_FAILURE);
        }
        herr = cudaHostAlloc((void **)&gpubuf->sequence_host[slot], sizeof(char) * batch_size * n_timesteps, cudaHostAllocDefault);
        if (herr != cudaSuccess) {
            OPENFISH_ERROR("cudaHostAlloc(sequence_host[%d]) failed: %s", slot, cudaGetErrorString(herr));
            exit(EXIT_FAILURE);
        }
        herr = cudaHostAlloc((void **)&gpubuf->qstring_host[slot], sizeof(char) * batch_size * n_timesteps, cudaHostAllocDefault);
        if (herr != cudaSuccess) {
            OPENFISH_ERROR("cudaHostAlloc(qstring_host[%d]) failed: %s", slot, cudaGetErrorString(herr));
            exit(EXIT_FAILURE);
        }
    }

    return gpubuf;
}

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
) {
    cudaFree(gpubuf->bwd_NTC);
    checkCudaError();
    cudaFree(gpubuf->post_NTC);
    checkCudaError();

    cudaFree(gpubuf->moves);
    checkCudaError();
    cudaFree(gpubuf->sequence);
    checkCudaError();
    cudaFree(gpubuf->qstring);
    checkCudaError();

    cudaFree(gpubuf->beam_vector);
    checkCudaError();
    cudaFree(gpubuf->states);
    checkCudaError();
    cudaFree(gpubuf->qual_data);
    checkCudaError();
    cudaFree(gpubuf->base_probs);
    checkCudaError();
    cudaFree(gpubuf->total_probs);
    checkCudaError();

    for (int slot = 0; slot < OPENFISH_HOST_RING; ++slot) {
        cudaFreeHost(gpubuf->moves_host[slot]);
        checkCudaError();
        cudaFreeHost(gpubuf->sequence_host[slot]);
        checkCudaError();
        cudaFreeHost(gpubuf->qstring_host[slot]);
        checkCudaError();
    }

    free(gpubuf);
}

/* Pick event-based timing when overlapping (stream != NULL); else host timers. */
#define OPENFISH_TIME_DECODE(stats_ptr, field, stream_handle, overlap, ...) \
    do { \
        if ((overlap) && (stats_ptr) != NULL) { \
            OPENFISH_TIME_SECTION_ASYNC((stats_ptr), field, (stream_handle), __VA_ARGS__); \
        } else { \
            OPENFISH_TIME_SECTION((stats_ptr), field, __VA_ARGS__); \
        } \
    } while (0)

/* Legacy path (stream == NULL): device-sync when collecting stats so per-phase
 * timers stay meaningful. Overlap path skips this via overlap_async. */
#define OPENFISH_PHASE_SYNC(stats_ptr) \
    do { \
        if ((stats_ptr) != NULL) { \
            cudaDeviceSynchronize(); \
            checkCudaError(); \
        } \
    } while (0)

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
    void *stream,
    int host_slot
) {
    const int num_states = pow(NUM_BASES, state_len);
    cudaStream_t s = (stream != NULL) ? (cudaStream_t)stream : (cudaStream_t)0;

    if (host_slot < 0 || host_slot >= OPENFISH_HOST_RING) {
        OPENFISH_ERROR("host_slot %d out of range [0, %d)", host_slot, OPENFISH_HOST_RING);
        exit(EXIT_FAILURE);
    }

    // calculate grid / block dims
    const int target_block_width = (int)ceil(sqrt((float)num_states));
    int block_width = 2;
    while (block_width < target_block_width) {
        block_width *= 2;
    }

    OPENFISH_LOG_TRACE("chosen block_dims: %d x %d for num_states %d", block_width, block_width, num_states);
    
    dim3 block_size(block_width, block_width, 1);
    dim3 block_size_beam(MAX_BEAM_WIDTH * NUM_BASES, 1, 1);
    dim3 block_size_gen(1, 1, 1);
	dim3 grid_size(batch_size, 1, 1);

    OPENFISH_LOG_TRACE("scores tensor dim (NTC): %d, %d, %d (host_slot=%d)", batch_size, n_timesteps, n_channels, host_slot);
    openfish_decode_stats_note_dims(stats, n_timesteps, batch_size, n_channels);

    scan_params_t scan_args = {0};
    scan_args.n_timesteps = n_timesteps;
    scan_args.batch_size = batch_size;
    scan_args.n_channels = n_channels;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;
    scan_args.score_scale = score_scale;

    // Reuse persistent pinned host ring slot; no per-call alloc/free.
    *moves = gpubuf->moves_host[host_slot];
    *sequence = gpubuf->sequence_host[host_slot];
    *qstring = gpubuf->qstring_host[host_slot];

    cudaMemsetAsync(gpubuf->moves, 0, sizeof(uint8_t) * batch_size * n_timesteps, s);
	checkCudaError();
    cudaMemsetAsync(gpubuf->sequence, 0, sizeof(char) * batch_size * n_timesteps, s);
	checkCudaError();
    cudaMemsetAsync(gpubuf->qstring, 0, sizeof(char) * batch_size * n_timesteps, s);
	checkCudaError();

    const int num_state_bits = (int)log2((double)num_states);
    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const float beam_cut = options->beam_cut;
    
    beam_params_t beam_args = {0};
    beam_args.n_timesteps = n_timesteps;
    beam_args.batch_size = batch_size;
    beam_args.n_channels = n_channels;
    beam_args.num_state_bits = num_state_bits;

    // the compact_offsets prefix-sum view overlays cand_scratch in the beam_search kernel,
    // so the int offsets must fit within the bool bloom-filter storage.
    ASSERT(MAX_BEAM_CANDIDATES * sizeof(int) <= HASH_PRESENT_BITS * sizeof(bool));

    // scores are read (and dequantized via score_scale) in three kernels; instantiate each on the
    // score element type. the f16 path passes score_scale = 1.0 and is numerically unchanged.
    // When stream != NULL (overlap), skip mid-phase syncs entirely so infer can run concurrently.
    const int overlap_async = (stream != NULL);

    OPENFISH_LOG_TRACE("bwd scan / beam search / fwd + post scan (score_dtype=%d)...", (int)score_dtype);
    if (score_dtype == OPENFISH_SCORE_I8) {
        OPENFISH_TIME_DECODE(stats, time_bwd_scan, s, overlap_async,
            bwd_scan<int8_t><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
            checkCudaError();
            if (!overlap_async) { OPENFISH_PHASE_SYNC(stats); }
        );

        OPENFISH_TIME_DECODE(stats, time_beam_search, s, overlap_async,
            beam_search<int8_t><<<grid_size,block_size_beam,num_states*sizeof(float),s>>>(
                beam_args, scores_NTC, gpubuf->bwd_NTC,
                (state_t *)gpubuf->states, gpubuf->moves, (beam_element_t *)gpubuf->beam_vector,
                beam_cut, fixed_stay_score, score_scale
            );
            checkCudaError();
            if (!overlap_async) { OPENFISH_PHASE_SYNC(stats); }
        );

        OPENFISH_TIME_DECODE(stats, time_fwd_post_scan, s, overlap_async,
            fwd_post_scan<int8_t><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
            checkCudaError();
            if (!overlap_async) { OPENFISH_PHASE_SYNC(stats); }
        );
    } else {
        OPENFISH_TIME_DECODE(stats, time_bwd_scan, s, overlap_async,
            bwd_scan<half><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
            checkCudaError();
            if (!overlap_async) { OPENFISH_PHASE_SYNC(stats); }
        );

        OPENFISH_TIME_DECODE(stats, time_beam_search, s, overlap_async,
            beam_search<half><<<grid_size,block_size_beam,num_states*sizeof(float),s>>>(
                beam_args, scores_NTC, gpubuf->bwd_NTC,
                (state_t *)gpubuf->states, gpubuf->moves, (beam_element_t *)gpubuf->beam_vector,
                beam_cut, fixed_stay_score, score_scale
            );
            checkCudaError();
            if (!overlap_async) { OPENFISH_PHASE_SYNC(stats); }
        );

        OPENFISH_TIME_DECODE(stats, time_fwd_post_scan, s, overlap_async,
            fwd_post_scan<half><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
            checkCudaError();
            if (!overlap_async) { OPENFISH_PHASE_SYNC(stats); }
        );
    }

    OPENFISH_LOG_TRACE("%s", "compute qual data...");
    OPENFISH_TIME_DECODE(stats, time_qual_data, s, overlap_async,
        compute_qual_data<<<grid_size,block_size_gen,0,s>>>(
            beam_args,
            gpubuf->post_NTC,
            (state_t *)gpubuf->states,
            gpubuf->qual_data,
            1.0f
        );
        checkCudaError();
        if (!overlap_async) { OPENFISH_PHASE_SYNC(stats); }
    );

    OPENFISH_LOG_TRACE("%s", "gen sequence...");
    OPENFISH_TIME_DECODE(stats, time_gen_sequence, s, overlap_async,
        generate_sequence<<<grid_size,block_size_gen,0,s>>>(
            beam_args,
            gpubuf->moves,
            (state_t *)gpubuf->states,
            gpubuf->qual_data,
            gpubuf->base_probs,
            gpubuf->total_probs,
            gpubuf->sequence,
            gpubuf->qstring,
            q_shift,
            q_scale
        );
        checkCudaError();
        if (!overlap_async) { OPENFISH_PHASE_SYNC(stats); }
    );

    OPENFISH_TIME_DECODE(stats, time_d2h_copy, s, overlap_async,
        cudaMemcpyAsync(*moves, gpubuf->moves, sizeof(uint8_t) * batch_size * n_timesteps, cudaMemcpyDeviceToHost, s);
        checkCudaError();
        cudaMemcpyAsync(*sequence, gpubuf->sequence, sizeof(char) * batch_size * n_timesteps, cudaMemcpyDeviceToHost, s);
        checkCudaError();
        cudaMemcpyAsync(*qstring, gpubuf->qstring, sizeof(char) * batch_size * n_timesteps, cudaMemcpyDeviceToHost, s);
        checkCudaError();
        /* Legacy / serial path: make host buffers valid before return.
         * Overlap path (stream != NULL): leave async — caller must sync the stream
         * before reading moves/sequence/qstring so infer can run concurrently. */
        if (stream == NULL) {
            cudaStreamSynchronize(s);
            checkCudaError();
        }
    );
}

void openfish_decode_free_host(
    uint8_t *moves,
    char *sequence,
    char *qstring
) {
    // No-op: host decode buffers are persistent and owned by openfish_gpubuf_t.
    (void)moves;
    (void)sequence;
    (void)qstring;
}

