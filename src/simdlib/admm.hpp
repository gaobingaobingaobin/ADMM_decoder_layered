#ifndef admm_namespace
#define admm_namespace

namespace admm
{
	typedef union __attribute__((__aligned__(32))) t_m256
	{
    	float   __emu_arr [8];
    	__m128  __emu_m128[2];
	} t_m256;
	
	typedef union __attribute__((__aligned__(32))) t_m256i
	{
    	int     i32b[8];
    	__m128i m128i[2];
    	__m256i m256i;
	} t_m256i;
	
	inline __m256i load(int* addr ){
		return _mm256_loadu_si256( (const __m256i*)addr );
	}

	inline __m256i load(unsigned int* addr ){
		return _mm256_loadu_si256( (const __m256i*)addr );
	}

	inline __m256 load(float* addr ){
		return _mm256_loadu_ps( addr );
	}

	inline void store(int* addr, __m256i value ){
		_mm256_store_si256( (__m256i*)addr, value );
	}

	inline void store(unsigned int* addr, __m256i value ){
		_mm256_store_si256( (__m256i*)addr, value );
	}

	inline void store(float* addr, __m256 value ){
		_mm256_store_ps( addr, value );
	}

	inline __m256 gather(float* addr, __m256i offsets ){
#if __AVX2__
		return _mm256_i32gather_ps (addr, offsets, 4);
#else
		t_m256i buf;
		buf.m256i = offsets;
		return _mm256_set_ps(    \
			addr[ buf.i32b[7] ], \
			addr[ buf.i32b[6] ], \
			addr[ buf.i32b[5] ], \
			addr[ buf.i32b[4] ], \
			addr[ buf.i32b[3] ], \
			addr[ buf.i32b[2] ], \
			addr[ buf.i32b[1] ], \
			addr[ buf.i32b[0] ]  \
		 );
#endif
	}

	inline unsigned int mask( __m256 value ){
		return _mm256_movemask_ps ( value );
	}

	inline __m256 min(const __m256 a, const __m256 b ){
		return _mm256_min_ps(a, b);
	}

	inline float min(const float a, const float b ){
		return (a<b) ? a : b;
	}

	inline float abs(const float a){
		return (a>0.0f) ? a : -a;
	}

	inline __m256 max(const __m256 a, const __m256 b ){
		return _mm256_max_ps(a, b);
	}

	inline float max(const float a, const float b ){
		return (a>b) ? a : b;
	}

	inline __m256 set(const float a){
		return _mm256_set1_ps(a);
	}

	inline __m256i set(int a){
		return _mm256_set1_epi32(a);
	}

	inline __m256i set(unsigned int a){
		return _mm256_set1_epi32(a);
	}

	inline __m256 cast(__m256i a){
		return _mm256_castsi256_ps(a);
	}

	inline __m256 select(__m256 a, __m256 b, __m256 c){
		return _mm256_blendv_ps(a, b, c);
	}

	inline __m256 select(__m256 a, __m256 b, __m256i c){
		const auto d = admm::cast(c);
		return _mm256_blendv_ps(a, b, d);
	}

	inline __m256 zero( ){
		return _mm256_setzero_ps( );
	}

	inline unsigned int count(__m256 a, int mask){
        const int value = admm::mask( a ) & mask;
        return _mm_popcnt_u32( value );
	}
	
	inline __m256 permute_8x32(__m256 a, __m256i b){
#if __AVX2__
		return 	_mm256_permutevar8x32_ps(a, b);
#else
		exit( 0 );
#endif
	}
}

#endif
