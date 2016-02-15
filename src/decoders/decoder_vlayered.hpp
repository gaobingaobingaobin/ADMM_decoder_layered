//=======================================
// december 7th, 2015
// Imen DEBBABI & Bertrand LEGAL
// Vertical Layered ADMM l2 LDPC Decoder
//======================================
//
#include "../simdlib/admm.hpp"

#define SKIP_PROJECTION true

void Decoder::ADMMDecoder_vlayered()
{
	int maxIter          = maxIteration;
	int numIter          = 0;
	const float rho      = 1.9f;//over relaxation
	const float un_m_rho = 1.0 - rho;
	float mu             = 3.0f;
	float alpha          = 0.8f;
	float tableau[12]    = { 0.0f};
	float tabCoef[12]    = { 0.0f };
	float llr[mBlocklength];

    if ((mBlocklength == 576) && (mNChecks == 288))
    {
			mu = 3.37309f; //penalty
    	tableau[2]  = 0.00001f; tabCoef[2] = 1.0f / ((2.0f - (2 * (tableau[2] / mu))));
			tableau[3]  = 2.00928f; tabCoef[3] = 1.0f / ((3.0f - (2 * (tableau[3] / mu))));
			tableau[6]  = 4.69438f; tabCoef[6] = 1.0f / ((6.0f - (2 * (tableau[6] / mu))));
    }
    else if ((mBlocklength == 2304) && (mNChecks == 1152))
    {
			mu = 3.81398683f; //penalty
	    tableau[2]  = 0.29669288f; tabCoef[2] = 1.0f / ((2.0f - (2 * (tableau[2] / mu))));
			tableau[3]  = 0.46964023f; tabCoef[3] = 1.0f / ((3.0f - (2 * (tableau[3] / mu))));
			tableau[6]  = 3.19548154f; tabCoef[6] = 1.0f / ((6.0f - (2 * (tableau[6] / mu))));
    }
    else if ((mBlocklength == 1152) && (mNChecks == 288))
    {
			mu = 3.43369433f; //penalty
      tableau[2]  = 0.00001f;    tabCoef[2] = 1.0f / ((2.0f - (2 * (tableau[2] / mu))));
			tableau[3]  = 1.01134229f; tabCoef[3] = 1.0f / ((3.0f - (2 * (tableau[3] / mu))));
			tableau[6]  = 4.43497372f; tabCoef[6] = 1.0f / ((6.0f - (2 * (tableau[6] / mu))));
    }
    else if((mBlocklength == 1944) && (mNChecks == 972) )//dv 2,3,4,11
    {
			mu          = 3.73378439f; //penalty
			tableau[2]  = 0.00001000f; tabCoef[2]  = 1.0f / ((2.0f  - (2 * (tableau[2]  / mu))));
			tableau[3]  = 1.73000336f; tabCoef[3]  = 1.0f / ((3.0f  - (2 * (tableau[3]  / mu))));
			tableau[4]  = 1.19197333f; tabCoef[4]  = 1.0f / ((4.0f  - (2 * (tableau[4]  / mu))));
			tableau[11] = 7.19659328f; tabCoef[11] = 1.0f / ((11.0f - (2 * (tableau[11] / mu))));
    }
   else
    {
			mu = 3.0f; //penalty
      tableau[2]  = 0.8f; tabCoef[2] = 1.0f / ((2.0f - (2 * (tableau[2] / mu))));
			tableau[3]  = 0.8f; tabCoef[3] = 1.0f / ((3.0f - (3 * (tableau[3] / mu))));
			tableau[6]  = 0.8f; tabCoef[6] = 1.0f / ((6.0f - (6 * (tableau[6] / mu))));
    }

	// initialize C2V msgs
   	 for (int i = 0;i < mPCheckMapSize; i++)
	{
		Lambda  [i] = 0.0f;
		zReplica[i] = 0.5f;
		latestProjVector[i] = 0.5f;
	}

	for (int i=0; i < mBlocklength ; i++)
	{
		const auto val = tableau[ VariableDegree[i] ];
		        llr[i] =  (_LogLikelihoodRatio[i] / mu) + (val/mu);
	}

	int ptr = 0;
	float* tskip = _LogLikelihoodRatio; // id
	//========= initialize V2C msgs
	for (int i=0; i < mBlocklength ; i++)
	{
		const int degVi = VariableDegree[ i ];
		const auto val  = tableau[ VariableDegree[i] ];
		float sum       = 0 ;

		for (int cnt = 0 ; cnt < VariableDegree[i] ; cnt++)
		{
			sum  +=  (zReplica[ t_row[ptr] ] + Lambda[ t_row[ptr]]) ;//msg_check_to_var
			ptr  +=1 ;
		}
		tskip[i]     = (sum - llr[i]);
		float  Li    = tabCoef[ degVi ] * tskip [i];
       OutputFromDecoder[ i ] = admm::max(admm::min(Li, 1.0f), 0.0f);
	}

	//========= start iterations
	int iter;
	int skipped = 0;
	int done = 0;

	for(iter = 0; iter < maxIter; iter++)
	{
	mIteration = iter + 1;

	float vector_before_proj[32] __attribute__((aligned(64)));
	int CumSumVarDegree = 0;
	int keep_cnt = 0; //position of the current VN to be updated

	//for each Vn
	for (int i=0; i < mBlocklength ; i++)
	{
		const int degVi     = VariableDegree[i];
		const auto val  = tableau[ VariableDegree[i] ];
		float sum       = 0.0f ;
		//========== msg c2v
		for (int j = 0 ; j < VariableDegree[i] ; j++)
		{
			const auto  ind = CumSumVarDegree + j;
			const auto dCN  = CheckDegree[ t_row1[ind] ];

			// precomputed before (in constructor)
			int ind_start = t_row_stride_8[ind];
#if SKIP_PROJECTION == true
			float diff = 0.0f;
#endif

			for (int cnt=0; cnt < dCN; cnt++ )
			{
				vector_before_proj[ cnt ] =  rho * OutputFromDecoder[t_col1[ind_start + cnt] ] + un_m_rho * zReplica[ ind_start + cnt] - Lambda[ind_start + cnt ];
				if (t_col1[ind_start + cnt] == i) // this is the only VN to be updated
					keep_cnt = cnt;
#if SKIP_PROJECTION == true
					diff    += admm::abs(latestProjVector[ind_start + cnt] - vector_before_proj[ cnt ]);
#endif
			}

#if SKIP_PROJECTION == true
			if( diff >= 0.0f) {
#else
			if( true ){
#endif

#if SKIP_PROJECTION == true
				for (int cnt=0; cnt < dCN; cnt++ )
				{
					const auto pos        = ind_start + cnt;
					latestProjVector[pos] = vector_before_proj[ cnt ];
				}
#endif
			// projection
			float *ztemp = mp.ProjectionInitiale (vector_before_proj, CheckDegree[ t_row1[ind] ]);
			//float *ztemp = mp.mProjectionPPd (vector_before_proj, CheckDegree[ t_row1[ind] ]);

			// update z and lambda only for the curent VN
			tskip[ i ]                        -= (zReplica[ ind_start + keep_cnt  ] + Lambda[ ind_start + keep_cnt ] );
			Lambda[ ind_start + keep_cnt ]     = Lambda[ ind_start + keep_cnt  ] + rho * (ztemp[keep_cnt] - OutputFromDecoder[ i]) + un_m_rho * (ztemp[keep_cnt] - zReplica[ ind_start + keep_cnt ]);
			zReplica[ ind_start + keep_cnt ]   = ztemp[keep_cnt];
			tskip[ i ]                        += (zReplica[ ind_start + keep_cnt  ] + Lambda[ ind_start + keep_cnt ] );
			}
		}

		// update output message from the current VN
		float  Li              = tabCoef[ degVi ] * tskip [ i ];//t [i] / ((degVi - (2 * (val / mu))));
        OutputFromDecoder[ i ] = admm::max(admm::min(Li ,  1.0f), 0.0f);
		CumSumVarDegree       += VariableDegree[i];
	}//end all VN

	//========== syndrom

	int CumSumCheckDegree = 0;
    int allVerified = 0;
	for (int j=0; j < mNChecks ; j++)
	{
		int syndrom = 0;
		for(int i = 0; i < CheckDegree[j]; i++)
		{
			const auto  ind = CumSumCheckDegree + i;
			int posVi       = t_col1[ind];
			syndrom        += (OutputFromDecoder[ posVi ]  - 0.5f) > 0.0f;
		}
		allVerified += ( syndrom & 0x01 );
		CumSumCheckDegree += CheckDegree[j];
	}
	//========== decision

	if(allVerified == 0)
	{
		mAlgorithmConverge = true;
		mValidCodeword     = true;
		break;
	}
	}//end all iterations


}//end function decoder_layered
