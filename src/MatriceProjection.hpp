/*
 * Decoder.hpp
 *
 *  Created on: 17 avr. 2015
 *      Author: legal
 *  Modified: june 2015
 */

//#define PERFORMANCE_ANALYSIS



#ifndef MatriceProjection_HPP_
#define MatriceProjection_HPP_

#include "./sorting/x86/SortDeg6.hpp"
#include "./sorting/x86/SortDeg7.hpp"
#include "./sorting/x86/SortDeg8.hpp"
//#include "./sorting/SortDeg14.hpp"
#include "./sorting/x86/SortDeg15.hpp"
#include "./sorting/x86/SortDegN.hpp"

#include "./simdlib/admm.hpp"


extern bool sort_compare_de(const NODE & a, const NODE & b){return (a.value > b.value);}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <class T = float, int N=32>
class MatriceProjection{
private:

	float results   [32]; // DECLARATION STATIQUE POUR GAGNER DU TEMPS !
	float results_0 [32] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // DECLARATION STATIQUE POUR GAGNER DU TEMPS !
	float results_1 [32] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // DECLARATION STATIQUE POUR GAGNER DU TEMPS !

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

private:
    long long int counter_1;
    long long int counter_2;
    long long int counter_3;
    long long int counter_4;
    long long int temps_1;
    long long int temps_2;
    long long int temps_3;
    long long int temps_4;

    __m256 invTab  [8];
    __m256 iClipTab[8];
    __m256 minusOne[8];

public:
	MatriceProjection()
	{
        counter_1 = 0;
        counter_2 = 0;
        counter_3 = 0;
        counter_4 = 0;
        temps_1   = 0;
        temps_2   = 0;
        temps_3   = 0;
        temps_4   = 0;

        minusOne[0] =  _mm256_set_ps (+0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, -1.0f);
        minusOne[1] =  _mm256_set_ps (+0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, -1.0f, -1.0f);
        minusOne[2] =  _mm256_set_ps (+0.0f, +0.0f, +0.0f, +0.0f, +0.0f, -1.0f, -1.0f, -1.0f);
        minusOne[3] =  _mm256_set_ps (+0.0f, +0.0f, +0.0f, +0.0f, -1.0f, -1.0f, -1.0f, -1.0f);
        minusOne[4] =  _mm256_set_ps (+0.0f, +0.0f, +0.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
        minusOne[5] =  _mm256_set_ps (+0.0f, +0.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
        minusOne[6] =  _mm256_set_ps (+0.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
        minusOne[7] =  _mm256_set_ps (-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);

        iClipTab[0] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000) );
        iClipTab[1] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000) );
        iClipTab[2] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000) );
        iClipTab[3] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000) );
        iClipTab[4] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000) );
        iClipTab[5] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000) );
        iClipTab[6] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000) );
        iClipTab[7] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000) );

        invTab[0] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000) );
        invTab[1] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000) );
        invTab[2] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000) );
        invTab[3] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000) );
        invTab[4] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000) );
        invTab[5] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000) );
        invTab[6] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000) );
        invTab[7] =  _mm256_castsi256_ps( _mm256_set_epi32 (0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000) );
	}

	~MatriceProjection()
	{

	}



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define OPTIMISATION false

	T* mProjectionPPd(float v[], int length)
	{
		if( length == 6 )
		{
#if OPTIMISATION
            // version optimisee de la projection
            __m256 rIn = _mm256_loadu_ps(v);
            __m256 rOu = projection_deg6( rIn );
            _mm256_storeu_ps(results, rOu);
            return results;
#else
            return mProjectionPPd<6>(v);
#endif
        } else if( length == 7 ) {
#if OPTIMISATION
            __m256 rIn = _mm256_loadu_ps(v);
            __m256 rOu = projection_deg7( rIn );
            _mm256_storeu_ps(results, rOu);
            return results;
#else
            return mProjectionPPd<7>(v);
#endif
        } else if( length == 8 ) {
#if OPTIMISATION == 2 // NOT WORKING ?
            __m256 rIn = _mm256_loadu_ps(v);
            __m256 rOu = projection_deg8( rIn );
            _mm256_storeu_ps(results, rOu);
            return results;
#else
            return mProjectionPPd<8>(v);
#endif
        } else if( length == 9 ) {
        	return mProjectionPPd<9>(v);

        } else if( length == 10 ) {
        	return mProjectionPPd<10>(v);

        } else if( length == 11 ) {
        	return mProjectionPPd<11>(v);

        } else if( length == 12 ) {
        	return mProjectionPPd<12>(v);

        } else if( length == 13 ) {
        	return mProjectionPPd<13>(v);

        } else if( length == 14 ) {
        	return mProjectionPPd<14>(v);

        } else if( length == 15 ) {
        	return mProjectionPPd<15>(v);

        } else if( length == 20 ) {
        	return mProjectionPPd<20>(v);

        } else {
                cout<<"(EE) ADMM not supported (length = " << length << ")" <<endl;
                exit ( 0 );
		}
		return NULL;
	}



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	template <int length=6> T* mProjectionPPd(float llr[])
	{
    int AllZero = 0;
		int AllOne  = 0;
		#pragma unroll
		for(int i = 0; i < length; i++)
		{
			AllZero = (llr[i] <= 0) ? AllZero + 1 : AllZero;
			AllOne  = AllOne  + (llr[i] >  1);
		}

		if(AllZero == length) // exit if the vector has all negative values, which means that the projection is all zero vector
		{
			return results_0;
		}

		if ( (AllOne==length) && ((length&0x01) == 0) ) // exit if the vector should be all one vector.
		{
			return results_1;
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		int zSorti[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		if     ( length ==  6 ) sort6_rank_order_reg (llr, zSorti);
		else if( length ==  7 ) sort7_rank_order_reg (llr, zSorti);
    else if( length ==  8 ) sort8_rank_order_reg (llr, zSorti);
    else if( length == 15 ) sort15_rank_order_reg(llr, zSorti);
    else function_sort<length>(llr, zSorti);



		int clip_idx = 0, zero_idx= 0;
		float constituent = 0;

		float llrClip[N];
		#pragma unroll
		for(int i = 0;i < length; i++)// project on the [0,1]^d cube
		{
			const auto tmp  = llr[i];
			const auto vMin = admm::max(tmp,  0.0f);
			const auto vMax = admm::min(vMin, 1.0f);
			llrClip[i]      = vMax;
			constituent    += vMax;//zClip[i].value;
		}

		int r = (int)constituent;
        r = r - (r & 0x01);

		// calculate constituent parity
		// calculate sum_Clip = $f_r^T z$
		float sum_Clip = 0;
		for(int i = 0; i < r+1; i++)
			sum_Clip += llrClip[i];

		for(int i = r + 1; i < length; i++)
			sum_Clip -= llrClip[i];

		if (sum_Clip <= r) // then done, return projection, beta = 0
		{
			#pragma unroll
			for(int i = 0; i < 8; i++)
				results[zSorti[i]] = llrClip[i];
			return results;
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		float beta     = 0;
		float beta_max = 0;
		if (r + 2 <= length)
			beta_max = (llr[r] - llr[r+1])/2; // assign beta_max
		else
			beta_max = llr[r];

		int   zSorti_m[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		float T_in[N];
        float T_out[N];
        int   order_out[N];

		for(int i = 0; i < r+1; i++)
            T_in[i] = llr[i] - 1.0f;

		for(int i = r+1; i < length; i++)
            T_in[i] = -llr[i];

		if     ( length ==  6 ) sort6_rank_order_reg_modif (T_in, T_out, zSorti_m, order_out);
    else if( length ==  7 ) sort7_rank_order_reg_modif (T_in, T_out, zSorti_m, order_out);
    else if( length ==  8 ) sort8_rank_order_reg_modif (T_in, T_out, zSorti_m, order_out);
    else if( length == 15 ) sort15_rank_order_reg_modif(T_in, T_out, zSorti_m, order_out);
    else function_sort<length>(T_in, T_out, zSorti_m, order_out);

		#pragma unroll
		for(int i = 0; i < length; i++)
		{
			if (llr[i] > 1)
				clip_idx++;
			if (llr[i] >= -1e-10)
				zero_idx++;
		}

		clip_idx--;

		int idx_start = 0;
		int idx_end   = 0;

        // les beta sont trié dans zBetaRep, il faut considérer uniquement ceux entre 0 et beta_max
		#pragma unroll
		for(int i = 0; i < length; i++)
		{
			if(T_out[i] < 1e-10 )
			{
				idx_start++;
			}
			if(T_out[i] < beta_max )
			{
				idx_end++;
			}
		}
		idx_end--;
		T active_sum = 0;

		#pragma unroll
		for(int i = 0;i < length; i++)
		{
			if(i > clip_idx && i <= r)
				active_sum += llr[i];
			if(i > r && i < zero_idx)
				active_sum -= llr[i];
		}

		T total_sum = 0;
		total_sum = active_sum + clip_idx + 1;

		int previous_clip_idx, previous_zero_idx;
		T previous_active_sum;
		bool change_pre     = true;

//		previous_clip_idx   = clip_idx;
//		previous_zero_idx   = zero_idx;
//		previous_active_sum = active_sum;


		for(int i = idx_start; i <= idx_end; i++)// pour tous les beta entre 0 et beta_max
		{
			if(change_pre)
			{
				// save previous things
				previous_clip_idx   = clip_idx;
				previous_zero_idx   = zero_idx;
				previous_active_sum = active_sum;
			}
			change_pre = false;

			beta = T_out[i];
			if(order_out[i] <= r)
			{
				clip_idx--;
				active_sum += llr[order_out[i]];
			}
			else
			{
				zero_idx++;
				active_sum -= llr[order_out[i]];
			}


			if (i < length - 1)
			{
				if (beta != T_out[i+1])
				{
					total_sum = (clip_idx+1) + active_sum - beta * (zero_idx - clip_idx - 1);
					change_pre = true;
					if(total_sum < r)
						break;
				}

			}
			else if (i == length - 1)
			{
				total_sum = (clip_idx + 1)  + active_sum - beta * (zero_idx - clip_idx - 1);
				change_pre = true;
			}
		}

		if (total_sum > r)
		{
			beta = -(r - clip_idx - 1 - active_sum)/(zero_idx - clip_idx - 1);
		}
		else
		{
			beta = -(r - previous_clip_idx - 1 - previous_active_sum)/(previous_zero_idx - previous_clip_idx - 1);
		}

		#pragma unroll
		for(int i = 0; i < length; i++)
		{
			const auto vA = llr[i];
			const auto vB = vA - beta;
			const auto vC = vA + beta;
			const auto vD = (i <= r) ? vB : vC;
			const auto vMin = std::max(vD,   0.0f);
			const auto vMax = std::min(vMin, 1.0f);
			results[zSorti[i]] = vMax;
		}
		return results;
	}


	//////////////////////////////////////////////////////////////////////////////


		float* ProjectionInitiale(float _v[], int length)
		{

			NODE   v[32];
			for(int i=0; i<length; i++)
			{
				v[i].value = _v[i];
				v[i].index =    i ;
			}

			NODE   zSort   [32]; // DECLARATION STATIQUE POUR GAGNER DU TEMPS !
			NODE   zClip   [32]; // DECLARATION STATIQUE POUR GAGNER DU TEMPS !
			NODE   zBetaRep[32]; // DECLARATION STATIQUE POUR GAGNER DU TEMPS !

			double zero_tol_offset = 1e-10;
			bool AllZero = true, AllOne = true;
			for(int i = 0; i < length; i++)
			{
				if(v[i].value > 0)
					AllZero = false;
				if(v[i].value <= 1)
					AllOne = false;
			}
			if(AllZero) // exit if the vector has all negative values, which means that the projection is all zero vector
			{
				for(int i = 0;i < length; i++)
					results[i] = 0;
				return results;
			}
			if (AllOne && (length%2 == 0)) // exit if the vector should be all one vector.
			{
				for(int i = 0;i < length; i++)
					results[i] = 1;
				return results;
			}

			for(int i = 0;i < length; i++) // keep the original indices while sorting,
			{
				zSort[i].index = v[i].index;
				zSort[i].value = v[i].value;
			}
			sort(zSort, zSort + length, sort_compare_de);//sort input vector, decreasing order


			int clip_idx = 0, zero_idx= 0;
			double constituent = 0;

			for(int i = 0;i < length; i++)// project on the [0,1]^d cube
			{
				zClip[i].value = min(max(zSort[i].value, 0.0),1.0);
				zClip[i].index = zSort[i].index;
				constituent += zClip[i].value;
			}
			int r = (int)floor(constituent);
			if (r & 1)
				r--;
			// calculate constituent parity

			// calculate sum_Clip = $f_r^T z$
			double sum_Clip = 0;
			for(int i = 0; i < r+1; i++)
				sum_Clip += zClip[i].value;
			for(int i = r + 1; i < length; i++)
				sum_Clip -= zClip[i].value;


			if (sum_Clip <= r) // then done, return projection, beta = 0
			{
				for(int i = 0; i < length; i++)
					results[zClip[i].index] = zClip[i].value;
				return results;
			}

			double beta = 0;
			double beta_max = 0;
			if (r + 2 <= length)
				beta_max = (zSort[r].value - zSort[r+1].value)/2; // assign beta_max
			else
				beta_max = zSort[r].value;

			// merge beta, save sort
			int left_idx = r, right_idx = r + 1;
			int count_idx = 0;
			while(count_idx < length)
			{
				double temp_a, temp_b;
				if(left_idx < 0)
				{
					while(count_idx < length)
					{
						zBetaRep[count_idx].index = right_idx;
						zBetaRep[count_idx].value = -zSort[right_idx].value;
						right_idx++;count_idx++;
					}
					break;
				}
				if(right_idx >= length)
				{
					while(count_idx < length)
					{
						zBetaRep[count_idx].index = left_idx;
						zBetaRep[count_idx].value = zSort[left_idx].value - 1;
						left_idx--;count_idx++;
					}
					break;
				}
				temp_a = zSort[left_idx].value - 1; temp_b = -zSort[right_idx].value;
				if(temp_a > temp_b || left_idx < 0)
				{
					zBetaRep[count_idx].index = right_idx;
					zBetaRep[count_idx].value = temp_b;
					right_idx++;count_idx++;
				}
				else
				{
					zBetaRep[count_idx].index = left_idx;
					zBetaRep[count_idx].value = temp_a;
					left_idx--;count_idx++;
				}
			}



			for(int i = 0; i < length; i++)
			{
				if (zSort[i].value > 1)
					clip_idx++;
				if (zSort[i].value >= 0 - zero_tol_offset)
					zero_idx++;
			}
			clip_idx--;
			bool first_positive = true, below_beta_max = true;
			int idx_start = 0, idx_end = 0;
			for(int i = 0;i < length; i++)
			{
				if(zBetaRep[i].value < 0 + zero_tol_offset && first_positive)
				{
					idx_start++;
				}
				if(zBetaRep[i].value < beta_max && below_beta_max)
				{
					idx_end++;
				}
			}
			idx_end--;
			double active_sum = 0;
			for(int i = 0;i < length; i++)
			{
				if(i > clip_idx && i <= r)
					active_sum += zSort[i].value;
				if(i > r && i < zero_idx)
					active_sum -= zSort[i].value;
			}
			double total_sum = 0;
			total_sum = active_sum + clip_idx + 1;

			int previous_clip_idx, previous_zero_idx;
			double previous_active_sum;
			bool change_pre = true;
			previous_clip_idx = clip_idx;
			previous_zero_idx = zero_idx;
			previous_active_sum = active_sum;
			for(int i = idx_start; i <= idx_end; i++)
			{
				if(change_pre)
				{
					// save previous things
					previous_clip_idx = clip_idx;
					previous_zero_idx = zero_idx;
					previous_active_sum = active_sum;
				}
				change_pre = false;

				beta = zBetaRep[i].value;
				if(zBetaRep[i].index <= r)
				{
					clip_idx--;
					active_sum += zSort[zBetaRep[i].index].value;
				}
				else
				{
					zero_idx++;
					active_sum -= zSort[zBetaRep[i].index].value;
				}
				if (i < length - 1)
				{
					if (beta != zBetaRep[i + 1].value)
					{
						total_sum = (clip_idx+1) + active_sum - beta * (zero_idx - clip_idx - 1);
						change_pre = true;
						if(total_sum < r)
							break;
					}
					else
					{
						if(0)
							cout<<"here here"<<endl;
					}
				}
				else if (i == length - 1)
				{
					total_sum = (clip_idx + 1)  + active_sum - beta * (zero_idx - clip_idx - 1);
					change_pre = true;
				}
			}
			if (total_sum > r)
			{
				beta = -(r - clip_idx - 1 - active_sum)/(zero_idx - clip_idx - 1);
			}
			else
			{
				beta = -(r - previous_clip_idx - 1 - previous_active_sum)/(previous_zero_idx - previous_clip_idx - 1);
			}
			for(int i = 0;i < length; i++)
			{
				if (i <= r)
				{
					results[zSort[i].index] = min(max(zSort[i].value - beta, 0.0),1.0);
				}
				else
				{
					results[zSort[i].index] = min(max(zSort[i].value + beta, 0.0),1.0);
				}

			}
			return results;
		}


};

#endif /* DECODER_HPP_ */
