//==========================================
// october 20th, 2015
// Imen DEBBABI & Bertrand LEGAL
// Horizontal Layered ADMM l2 LDPC Decoder
//==========================================

#include "../simdlib/admm.hpp"

#define SKIP_PROJECTION true

void Decoder::ADMMDecoder_hlayered()
{
	int maxIter          = maxIteration;
	const float rho      = 1.9f;//over relaxation
	const float un_m_rho = 1.0 - rho;
	float mu             = 3.0f;
	float tableau[12]    = { 0.0f };
	float tabCoef[12]    = { 0.0f };

    if ((mBlocklength == 576) && (mNChecks == 288))
    {
		mu          = 3.37309f; //penalty
    	tableau[2]  = 0.00001f; tabCoef[2] = 1.0f / ((2.0f - (2 * (tableau[2] / mu))));
		tableau[3]  = 2.00928f; tabCoef[3] = 1.0f / ((3.0f - (2 * (tableau[3] / mu))));
		tableau[6]  = 4.69438f; tabCoef[6] = 1.0f / ((6.0f - (2 * (tableau[6] / mu))));
    }
    else if ((mBlocklength == 2304) && (mNChecks == 1152))
    {
		mu          = 3.81398683f; //penalty
        tableau[2]  = 0.29669288f; tabCoef[2] = 1.0f / ((2.0f - (2 * (tableau[2] / mu))));
		tableau[3]  = 0.46964023f; tabCoef[3] = 1.0f / ((3.0f - (2 * (tableau[3] / mu))));
		tableau[6]  = 3.19548154f; tabCoef[6] = 1.0f / ((6.0f - (2 * (tableau[6] / mu))));
    }
    else if ((mBlocklength == 1152) && (mNChecks == 288))
    {
		mu          = 3.43369433f; //penalty
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
		mu          = 3.0f; //penalty
        tableau[2]  = 0.8f; tabCoef[2] = 1.0f / ((2.0f - (2 * (tableau[2] / mu))));
		tableau[3]  = 0.8f; tabCoef[3] = 1.0f / ((3.0f - (3 * (tableau[3] / mu))));
		tableau[6]  = 0.8f; tabCoef[6] = 1.0f / ((6.0f - (6 * (tableau[6] / mu))));
    }


   	for (int i = 0;i < mPCheckMapSize; i++)
	{
		Lambda  [i]         = 0.0f;
		zReplica[i]         = 0.5f;
		latestProjVector[i] = 0.5f;
	}

	int ptr = 0;
	float* t = _LogLikelihoodRatio;

	for (int i=0; i < mBlocklength ; i++)
	{
		float sum = 0 ;
		const auto val = tableau[ VariableDegree[i] ];
		for (int cnt = 0 ; cnt < VariableDegree[i] ; cnt++)
		{
			sum  +=  (zReplica[ t_row[ptr] ] + Lambda[ t_row[ptr]]) ;//msg_check_to_var
			ptr  +=1 ;
		}
		t[i] = sum - (_LogLikelihoodRatio[i] / mu) - (val/mu);//tableau[i] / mu ;
	}

	//========= start iterations
	for(int iter = 0; iter < maxIter; iter++)
	{
		mIteration = iter + 1;
		int CumSumCheckDegree = 0;
		float v_before_proj[32] __attribute__((aligned(64)));

		//for each Cn
		for (int j=0; j < mNChecks ; j++)
		{
#if SKIP_PROJECTION == true
			float difference = 0.0f;
#endif
			const int degCn = CheckDegree[j];
			for(int i=0; i< degCn; i++)
			{
				const auto  ind            = CumSumCheckDegree + i;
				const int posVi            = t_col1[ind]; // position of 1 in the row
				const int degVi            = VariableDegree[ posVi ];
				const float Li 	           = tabCoef[ degVi ] * t [ posVi ];
				const float vmin           = admm::min(Li ,  1.0f);
				OutputFromDecoder[ posVi ] = admm::max(vmin, 0.0f);
				v_before_proj[i]           = rho * OutputFromDecoder[ posVi ] + un_m_rho * zReplica[ind]- Lambda[ind];
#if SKIP_PROJECTION == true
				difference                += admm::abs(latestProjVector[ind] - v_before_proj[i]);
#endif
			}

#if SKIP_PROJECTION == true
			if( difference > 1.0e-4 ){
#else
			if( true ){
#endif
				float* ztemp  = mp.ProjectionInitiale(v_before_proj, CheckDegree[j]);
				//float *ztemp = mp.mProjectionPPd (v_before_proj, degCn);
				for(int i = 0; i < degCn; i++)
				{
					const auto  ind       = CumSumCheckDegree + i;
					const int posVi       = t_col1[ind];
					t[posVi]             -= (zReplica[ind] + Lambda[ind]); // BLG
					Lambda[ind]           = Lambda[ind] + rho * (ztemp[i] - OutputFromDecoder[posVi]) + un_m_rho * (ztemp[i] - zReplica[ind]);
					zReplica[ind]         = ztemp[i];
					t[posVi]             += (zReplica[ind] + Lambda[ind]);
#if SKIP_PROJECTION == true
					latestProjVector[ind] = v_before_proj[i];
#endif
				}
			}
			CumSumCheckDegree += degCn;
		}//end all Cn

		CumSumCheckDegree = 0;
		int allVerified = 0;
		for (int j=0; j < mNChecks ; j++)
		{
			int syndrom = 0;
			const int degCn = CheckDegree[j];
			for(int i = 0; i < degCn; i++)
			{
				const auto  ind = CumSumCheckDegree + i;
				int posVi       = t_col1[ind];
				syndrom        += (OutputFromDecoder[ posVi ]  > 0.5);
			}
			allVerified       += ( syndrom & 0x01 );
			CumSumCheckDegree +=  degCn;
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

#undef SKIP_PROJECTION
