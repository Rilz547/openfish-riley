/** Riley Updates (Remove at the end)
 * @file decode_hip.c
 * @lastmodified: NULL-init gpubuf host buffer fields (CUDA owns persistent pinned hosts; HIP still mallocs per decode).
 * @lastpatched: 2026-07-18

******************************************************************************/

#include <openfish/openfish.h>
#include "openfish_defs.h"
#include "scan_hip.h"
#include "beam_search_hip.h"
#include "error.h"
#include "hip_utils.h"
#include "misc.h"

#include <openfish/openfish_error.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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

#define OPENFISH_TIME_SECTION(stats_ptr, field, ...) \
    do { \
        double _openfish_t0 = 0.0; \
        double _openfish_t1 = 0.0; \
        hipError_t _openfish_ret = hipSuccess; \
        if ((stats_ptr) != NULL) { \
            _openfish_t0 = openfish_realtime(); \
        } \
        __VA_ARGS__ \
        if ((stats_ptr) != NULL) { \
            _openfish_t1 = openfish_realtime(); \
            (stats_ptr)->field += _openfish_t1 - _openfish_t0; \
        } \
        (void)_openfish_ret; \
    } while (0)


openfish_gpubuf_t *openfish_gpubuf_init(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    hipError_t ret;
    openfish_gpubuf_t *gpubuf = (openfish_gpubuf_t *)(malloc(sizeof(openfish_gpubuf_t)));
    MALLOC_CHK(gpubuf);

    const int num_states = pow(NUM_BASES, state_len);

    // scan tensors
    ret = hipMalloc((void **)&gpubuf->bwd_NTC, sizeof(float) * batch_size * (n_timesteps + 1) * num_states);
	checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->post_NTC, sizeof(float) * batch_size * (n_timesteps + 1) * num_states);
	checkHipError(); HIP_CHECK(ret);

    // return buffers
    ret = hipMalloc((void **)&gpubuf->moves, sizeof(uint8_t) * batch_size * n_timesteps);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->sequence, sizeof(char) * batch_size * n_timesteps);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->qstring, sizeof(char) * batch_size * n_timesteps);
    checkHipError(); HIP_CHECK(ret);

    // beamsearch buffers
    ret = hipMalloc((void **)&gpubuf->beam_vector, sizeof(beam_element_t) * batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1));
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->states, sizeof(state_t) * batch_size * n_timesteps);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->qual_data, sizeof(float) * batch_size * n_timesteps * NUM_BASES);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->base_probs, sizeof(float) * batch_size * n_timesteps);
    checkHipError(); HIP_CHECK(ret);
    ret = hipMalloc((void **)&gpubuf->total_probs, sizeof(float) * batch_size * n_timesteps);
    checkHipError(); HIP_CHECK(ret);

    /* CUDA allocates persistent pinned hosts; HIP leaves these unused. */
    gpubuf->moves_host = NULL;
    gpubuf->sequence_host = NULL;
    gpubuf->qstring_host = NULL;

    return gpubuf;
}

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
) {
    hipError_t ret;
    ret = hipFree(gpubuf->bwd_NTC);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->post_NTC);
    checkHipError(); HIP_CHECK(ret);

    ret = hipFree(gpubuf->moves);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->sequence);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->qstring);
    checkHipError(); HIP_CHECK(ret);

    ret = hipFree(gpubuf->beam_vector);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->states);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->qual_data);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->base_probs);
    checkHipError(); HIP_CHECK(ret);
    ret = hipFree(gpubuf->total_probs);
    checkHipError(); HIP_CHECK(ret);

    free(gpubuf);
}

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
    void *stream
) {
    (void)stream; /* CUDA overlap streams; HIP stream path not wired yet */
    hipError_t ret;
    const int num_states = pow(NUM_BASES, state_len);

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

    OPENFISH_LOG_TRACE("scores tensor dim (NTC): %d, %d, %d", batch_size, n_timesteps, n_channels);
    openfish_decode_stats_note_dims(stats, n_timesteps, batch_size, n_channels);

    scan_params_t scan_args = {0};
    scan_args.n_timesteps = n_timesteps;
    scan_args.batch_size = batch_size;
    scan_args.n_channels = n_channels;
    scan_args.num_states = num_states;
    scan_args.fixed_stay_score = options->blank_score;
    scan_args.score_scale = score_scale;

    // init results
    *moves = (uint8_t *)malloc(batch_size * n_timesteps * sizeof(uint8_t));
    MALLOC_CHK(*moves);
    *sequence = (char *)malloc(batch_size * n_timesteps * sizeof(char));
    MALLOC_CHK(*sequence);
    *qstring = (char *)malloc(batch_size * n_timesteps * sizeof(char));
    MALLOC_CHK(*qstring);

    ret = hipMemset(gpubuf->moves, 0, sizeof(uint8_t) * batch_size * n_timesteps);
	checkHipError(); HIP_CHECK(ret);
    ret = hipMemset(gpubuf->sequence, 0, sizeof(char) * batch_size * n_timesteps);
	checkHipError(); HIP_CHECK(ret);
    ret = hipMemset(gpubuf->qstring, 0, sizeof(char) * batch_size * n_timesteps);
	checkHipError(); HIP_CHECK(ret);

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

    // bwd scan
    // fwd + post scan
    // beam search

    // the compact_offsets prefix-sum view overlays cand_scratch in the beam_search kernel,
    // so the int offsets must fit within the bool bloom-filter storage.
    ASSERT(MAX_BEAM_CANDIDATES * sizeof(int) <= HASH_PRESENT_BITS * sizeof(bool));

    // scores are read (and dequantized via score_scale) in three kernels; instantiate each on the
    // score element type. the f16 path passes score_scale = 1.0 and is numerically unchanged.
    OPENFISH_LOG_TRACE("bwd scan / beam search / fwd + post scan (score_dtype=%d)...", (int)score_dtype);
    if (score_dtype == OPENFISH_SCORE_I8) {
        OPENFISH_TIME_SECTION(stats, time_bwd_scan,
            bwd_scan<int8_t><<<grid_size,block_size>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
            checkHipError();
            ret = hipDeviceSynchronize();
            checkHipError(); HIP_CHECK(ret);
        );

        OPENFISH_TIME_SECTION(stats, time_beam_search,
            beam_search<int8_t><<<grid_size,block_size_beam,num_states*sizeof(float)>>>(
                beam_args, scores_NTC, gpubuf->bwd_NTC,
                (state_t *)gpubuf->states, gpubuf->moves, (beam_element_t *)gpubuf->beam_vector,
                beam_cut, fixed_stay_score, score_scale
            );
            checkHipError();
            ret = hipDeviceSynchronize();
            checkHipError(); HIP_CHECK(ret);
        );

        OPENFISH_TIME_SECTION(stats, time_fwd_post_scan,
            fwd_post_scan<int8_t><<<grid_size,block_size>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
            checkHipError();
            ret = hipDeviceSynchronize();
            checkHipError(); HIP_CHECK(ret);
        );
    } else {
        OPENFISH_TIME_SECTION(stats, time_bwd_scan,
            bwd_scan<half><<<grid_size,block_size>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
            checkHipError();
            ret = hipDeviceSynchronize();
            checkHipError(); HIP_CHECK(ret);
        );

        OPENFISH_TIME_SECTION(stats, time_beam_search,
            beam_search<half><<<grid_size,block_size_beam,num_states*sizeof(float)>>>(
                beam_args, scores_NTC, gpubuf->bwd_NTC,
                (state_t *)gpubuf->states, gpubuf->moves, (beam_element_t *)gpubuf->beam_vector,
                beam_cut, fixed_stay_score, score_scale
            );
            checkHipError();
            ret = hipDeviceSynchronize();
            checkHipError(); HIP_CHECK(ret);
        );

        OPENFISH_TIME_SECTION(stats, time_fwd_post_scan,
            fwd_post_scan<half><<<grid_size,block_size>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
            checkHipError();
            ret = hipDeviceSynchronize();
            checkHipError(); HIP_CHECK(ret);
        );
    }

    OPENFISH_LOG_TRACE("%s", "compute qual data...");
    OPENFISH_TIME_SECTION(stats, time_qual_data,
        compute_qual_data<<<grid_size,block_size_gen>>>(
            beam_args,
            gpubuf->post_NTC,
            (state_t *)gpubuf->states,
            gpubuf->qual_data,
            1.0f
        );
        checkHipError();
        ret = hipDeviceSynchronize();
        checkHipError(); HIP_CHECK(ret);
    );

    OPENFISH_LOG_TRACE("%s", "gen sequence...");
    OPENFISH_TIME_SECTION(stats, time_gen_sequence,
        generate_sequence<<<grid_size,block_size_gen>>>(
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
        checkHipError();
        ret = hipDeviceSynchronize();
        checkHipError(); HIP_CHECK(ret);
    );

    OPENFISH_TIME_SECTION(stats, time_d2h_copy,
        ret = hipMemcpy(*moves, gpubuf->moves, sizeof(uint8_t) * batch_size * n_timesteps, hipMemcpyDeviceToHost);
        checkHipError(); HIP_CHECK(ret);
        ret = hipMemcpy(*sequence, gpubuf->sequence, sizeof(char) * batch_size * n_timesteps, hipMemcpyDeviceToHost);
        checkHipError(); HIP_CHECK(ret);
        ret = hipMemcpy(*qstring, gpubuf->qstring, sizeof(char) * batch_size * n_timesteps, hipMemcpyDeviceToHost);
        checkHipError(); HIP_CHECK(ret);
    );
}

void openfish_decode_free_host(
    uint8_t *moves,
    char *sequence,
    char *qstring
) {
    free(moves);
    free(sequence);
    free(qstring);
}

void openfish_decode_stats_finish(openfish_decode_stats_t *stats) {
    (void)stats;
}

