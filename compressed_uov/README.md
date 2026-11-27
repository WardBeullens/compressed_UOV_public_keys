  
# UOV signature

This project contains the reference and optimized implementations of the Unbalanced Oil and Vinegar (UOV) signature system.

## Licence

Public Domain (https://creativecommons.org/share-your-work/public-domain/cc0/)
unless stated differently at the top of each file.

## Parameters

|         |NIST Sec. Level | Parameter (n,m,GF(q)) | epk size (bytes) | esk size (bytes) | cpk size (bytes)   | csk size (bytes)   | signature size (bytes) |
|:--------|:--------:|:-------------:|----------:|---------:|---------:|----------|----------:|
| uov-Ip  |1         |119,45,GF(256) | 321,300   |278,087   |46,591     | 32      |  135      |
| uov-Is  |1         |160,64,GF(16)  | 412,160   |348,704   |66,576     | 32      |   96      |
| uov-III |3         |193,72,GF(256) | 1,347,912 |1,167,440 |189,232    | 32      |  209      |
| uov-V   |5         |259,96,GF(256) | 3,232,320 |2,801,024 |446,992    | 32      |  275      |


## Seed-derived entries of P3 (this fork)

The `*_pkc` variants use a smaller compressed public key. Key generation chooses the oil
matrix O column by column, by a Gram-Schmidt-like procedure, so that the first
min(j*m, v) coordinates of column j of P3 equal values expanded from pk_seed right after
P1 and P2; the public key leaves those out (`ov_seeded_p3.c`). O is then no longer the
hash output of sk_seed, so it is taken from the secret key, which the pkc format stores
anyway. Signing and verification are unchanged.

|         | cpk size (bytes), standard pkc | cpk size (bytes), this fork |
|:--------|---------:|---------:|
| uov-Is  | 66,576   | 63,568 (-4.5%) |
| uov-Ip  | 46,591   | 43,364 (-6.9%) |
| uov-III | 189,232  | 180,690 (-4.5%) |
| uov-V   | 446,992  | 431,574 (-3.4%) |

The classic and pkc_skc variants are unchanged. Compiling with `-D_OV_NO_SEEDED_P3`
(e.g. `make "CC=gcc -D_OV_NO_SEEDED_P3" PROJ=Ip_pkc`) gives the standard pkc format, which
reproduces upstream's KATs. `make PROJ=<proj> seeded_p3-test` checks the seeded entries of
freshly generated keys, and `./check_seeded_kat.sh` regenerates the KATs of the four
seeded pkc variants and compares them with `seeded_kat.sha256`. Note that passing
`CFLAGS=...` to make replaces the Makefile's own flags, including `-D_NIST_KAT_`; pass
extra defines through `CC` instead.

## Instructions for testing and generating KATs

Type **make**    
```
make
```
It will generate an executable **sign_api-test** testing API functions (crypto_keygen(), crypto_sign(), and crypto_verify()).  

### **Options for Parameters:**

The source code of all supporting parameters is stored in subdirectories of the parameter names.
The default setting is compiling **Ip**.  
We use **PROJ** parameters for compiling other parameters.  
For example:  
```
make PROJ=Is_pkc
```
```
make PROJ=III_pkc_skc
```

### **Generating KAT:**
Use the following script:  
```
source ./generate_KAT.sh
```

