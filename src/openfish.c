/** Riley Updates (Remove at the end)
 * @file openfish.c
 * @lastmodified: openfish_gpubuf_size now includes the three persistent pinned host decode buffers.
 * @lastpatched: 2026-07-18

******************************************************************************/

#include <openfish/openfish.h>
#include "openfish_defs.h"

#include <string.h>

void openfish_decode_stats_reset(openfish_decode_stats_t *stats) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
}

openfish_opt_t openfish_decoder_default_opts(void) {
    openfish_opt_t opt = {100.0f, 2.0f, 0.0f, 1.0f};
    return opt;
}

size_t openfish_gpubuf_size(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    const size_t num_states = (size_t)1 << (2 * state_len);
    return
        sizeof(float) * (size_t)batch_size * (n_timesteps + 1) * num_states +          // bwd_NTC
        sizeof(float) * (size_t)batch_size * (n_timesteps + 1) * num_states +          // post_NTC
        sizeof(uint8_t) * (size_t)batch_size * n_timesteps +                            // moves (device)
        sizeof(char) * (size_t)batch_size * n_timesteps +                               // sequence (device)
        sizeof(char) * (size_t)batch_size * n_timesteps +                               // qstring (device)
        sizeof(beam_element_t) * (size_t)batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1) + // beam_vector
        sizeof(state_t) * (size_t)batch_size * n_timesteps +                            // states
        sizeof(float) * (size_t)batch_size * n_timesteps * NUM_BASES +                  // qual_data
        sizeof(float) * (size_t)batch_size * n_timesteps +                              // base_probs
        sizeof(float) * (size_t)batch_size * n_timesteps +                              // total_probs
        sizeof(uint8_t) * (size_t)batch_size * n_timesteps +                            // moves_host (pinned)
        sizeof(char) * (size_t)batch_size * n_timesteps +                               // sequence_host (pinned)
        sizeof(char) * (size_t)batch_size * n_timesteps;                                // qstring_host (pinned)
}
