// SPDX-License-Identifier: CC0 OR Apache-2.0
/// @file ov_seeded_p3.h
/// @brief Seed-derived entries of P3 (the _OV_SEEDED_P3 variant of the pkc format).
///
/// Key generation chooses the oil matrix O column by column so that the first
/// n_j = min(j*m, v) coordinates of column j of P3, ordered by row and then by equation,
/// equal targets expanded from pk_seed right after P1 and P2. The public key leaves them
/// out. Entry (i,j), i < j, of P3 is  L_i^T o_j + o_i^T P2 e_j  with
/// L_i = (P1 + P1^T) o_i + P2 e_i, so each column is a linear system in o_j once the earlier
/// columns are fixed. Column 1 is a uniformly random solution of its m x v system, and
/// columns 2..o-1 all solve the same v x v system (see ov_match_P3_targets).
///
#ifndef _OV_SEEDED_P3_H_
#define _OV_SEEDED_P3_H_

#include "params.h"
#include "ov_keypair.h"
#include "utils_prng.h"

#if defined(_OV_SEEDED_P3)

/// targets of P3, as a vector of N_SEEDED field elements
#define _SEEDED_P3_BYTE  ((_GFSIZE == 16) ? (N_SEEDED + 1) / 2 : N_SEEDED)

/// @brief Overwrites columns 1..o-1 of O so that the seeded entries of P3 equal the targets.
/// @return 0 on success, -1 if a system does not have full rank (key generation then picks
/// another seed).
#define ov_match_P3_targets PQOV_NAMESPACE(ov_match_P3_targets)
int ov_match_P3_targets( unsigned char *O, const unsigned char *P1, const unsigned char *P2,
                         const unsigned char *targets );

/// @brief P3 without its seeded entries.
#define ov_pack_P3 PQOV_NAMESPACE(ov_pack_P3)
void ov_pack_P3( unsigned char *packed, const unsigned char *P3 );

/// @brief inverse of ov_pack_P3, with the seeded entries taken from the targets.
#define ov_unpack_P3 PQOV_NAMESPACE(ov_unpack_P3)
void ov_unpack_P3( unsigned char *P3, const unsigned char *packed, const unsigned char *targets );

#endif

#endif
