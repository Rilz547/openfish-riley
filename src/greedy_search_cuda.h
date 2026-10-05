#ifndef GREEDY_SEARCH_CUDA_H
#define GREEDY_SEARCH_CUDA_H

#include <math.h>
#include <float.h>
#include <cuda_fp16.h>

#include "openfish_defs.h"
#include "scan_cuda.h"

// Width-1 ("greedy") replacement for beam_search, used as the cheap draft in speculative decode.
//
// Walks one path through the CRF per chunk. At every timestep the five transitions out of the current
// state (stay, or step to one of the four successor k-mers) are scored exactly as in beam_search --
// emission score + back guide of the destination state, with a fixed score for stays -- and the best one
// is taken. There is no beam, no hash merging of stays/steps and no pruning, so the whole chunk is a
// single sequential walk.
//
// Output format is identical to beam_search (states[t] = k-mer after step t, moves[t] = 1 for a step and
// 0 for a stay, moves[0] = 1), so compute_qual_data / generate_sequence consume it unchanged.
//
// margins[chunk] is the mean over timesteps of (best - second best) transition score, i.e. how decisively
// the walk preferred its choice. Low values flag chunks where the greedy path is unreliable.
template<typename ScoreT>
__global__ void greedy_search(
    const beam_params_t args,
    const void *_scores_NTC,
    const float *_bwd_NTC,
    state_t *_states,
    uint8_t *_moves,
    float *_margins,
    const float fixed_stay_score,
    const float score_scale
) {
    const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
    if (chunk >= args.batch_size || threadIdx.x != 0 || threadIdx.y != 0) {
        return;
    }

    const uint64_t n_timesteps = args.n_timesteps;
    const uint64_t n_channels = args.n_channels;
    const int num_state_bits = args.num_state_bits;
    const size_t num_states = 1ull << num_state_bits;
    const state_t states_mask = (state_t)(num_states - 1);

    const ScoreT *scores_NTC = (const ScoreT *)_scores_NTC + chunk * n_timesteps * (num_states * NUM_BASES);
    const float *bwd_NTC = _bwd_NTC + chunk * num_states * (n_timesteps + 1);
    state_t *states = _states + chunk * n_timesteps;
    uint8_t *moves = _moves + chunk * n_timesteps;

    // seed with the state the back guide likes best at the start of the chunk
    state_t state = 0;
    float seed_score = bwd_NTC[0];
    for (size_t s = 1; s < num_states; ++s) {
        if (bwd_NTC[s] > seed_score) {
            seed_score = bwd_NTC[s];
            state = (state_t)s;
        }
    }

    double margin_sum = 0.0;

    for (uint64_t t = 0; t < n_timesteps; ++t) {
        const ScoreT *const block_scores = scores_NTC + t * n_channels;
        const float *const block_back_scores = bwd_NTC + ((t + 1) << num_state_bits);

        float best = -FLT_MAX;
        float second = -FLT_MAX;
        state_t best_state = state;
        uint8_t best_move = 0;

        // steps: shift the k-mer left and append the new base (same indexing as beam_search)
        for (int base = 0; base < NUM_BASES; ++base) {
            const state_t new_state = (state_t)(((state << NUM_BASE_BITS) & states_mask) | (state_t)base);
            const state_t move_idx = (state_t)((new_state << NUM_BASE_BITS) + ((state << NUM_BASE_BITS) >> num_state_bits));
            const float sc = load_score(block_scores[move_idx]) * score_scale + block_back_scores[new_state];
            if (sc > best) {
                second = best;
                best = sc;
                best_state = new_state;
                best_move = 1;
            } else if (sc > second) {
                second = sc;
            }
        }

        // stay: same k-mer, fixed score
        {
            const float sc = fixed_stay_score + block_back_scores[state];
            if (sc > best) {
                second = best;
                best = sc;
                best_state = state;
                best_move = 0;
            } else if (sc > second) {
                second = sc;
            }
        }

        state = best_state;
        states[t] = state;
        moves[t] = best_move;
        margin_sum += (double)(best - second);
    }

    moves[0] = 1;  // always step in the first event
    _margins[chunk] = (n_timesteps > 0) ? (float)(margin_sum / (double)n_timesteps) : 0.0f;
}

#endif // GREEDY_SEARCH_CUDA_H
