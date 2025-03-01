# Common Tools



```c++
static inline uint32_t rotl32(uint32_t n, unsigned int c)
{
    const unsigned int mask = 31;

    c &= mask;
    unsigned int neg_c = (unsigned int)(-(int)c);
    return (n << c) | (n >> (neg_c & mask));
}

static inline uint32_t rotr32(uint32_t n, unsigned int c)
{
    const unsigned int mask = 31;

    c &= mask;
    unsigned int neg_c = (unsigned int)(-(int)c);
    return (n >> c) | (n << (neg_c & mask));
}

static inline uint32_t clz32(uint32_t x)
{
    return x ? (uint32_t)__builtin_clz(x) : 32;
}

static inline uint32_t popcount32(uint32_t x)
{
    return (uint32_t)__builtin_popcount(x);
}

static inline uint32_t mul_hi32(uint32_t x, uint32_t y)
{
    return (uint32_t)(((uint64_t)x * (uint64_t)y) >> 32);
}


/** FNV 32-bit prime. */
static const uint32_t fnv_prime = 0x01000193;

/** FNV 32-bit offset basis. */
static const uint32_t fnv_offset_basis = 0x811c9dc5;

/**
 * The implementation of FNV-1 hash.
 *
 * See https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1_hash.
 */
NO_SANITIZE("unsigned-integer-overflow")
static inline uint32_t fnv1(uint32_t u, uint32_t v) noexcept
{
    return (u * fnv_prime) ^ v;
}

/**
 * The implementation of FNV-1a hash.
 *
 * See https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1a_hash.
 */
NO_SANITIZE("unsigned-integer-overflow")
static inline uint32_t fnv1a(uint32_t u, uint32_t v) noexcept
{
    return (u ^ v) * fnv_prime;
}

```



**_builtin_popcount(x):** This function is used to count the number of one’s(set bits) in an integer.

**__builtin_clz(x):** This function is used to count the leading zeros of the integer. Note : clz = count leading zero’s
**Example:** It counts number of zeros before the first occurrence of one(set bit).

