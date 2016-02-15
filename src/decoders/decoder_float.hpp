#define SKIP_PROJECTION true

void Decoder::ADMMDecoder_float_l2()
{
	int maxIter          = maxIteration;
	const float rho      = 1.9f;//over relaxation
	const float un_m_rho = 1.0 - rho;
	float mu             = 3.0f;
	float tableau[16]    = { 0.0f };
	float tabCoef[16]    = { 0.0f };

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
//		printf("la...\n");
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

    // ON PRECALCUL LES COEFFICIENTS
    for (int j = 1; j < 16; j++) {
        tabCoef[j] = 1.0f / (((float)j - (2 * (tableau[j] / mu))));
        tableau[j] /= mu;
    }

    // ON RESCALE LES LLRs
    for (int j = 0; j < mBlocklength; j++) {
        _LogLikelihoodRatio[j] = _LogLikelihoodRatio[j] / mu;
    }

    for (int i = 0;i < mPCheckMapSize; i++) {
		Lambda  [i]         = 0.0f;
		zReplica[i]         = 0.5f;
#if SKIP_PROJECTION == true
		latestProjVector[i] = 0.5f;
#endif
    }

    for(int i = 0; i < maxIter; i++)
    {
        int ptr    = 0;
        mIteration = i + 1;

        // VN processing kernel
        for (int j = 0; j < mBlocklength; j++)
        {
            const auto degVi = VariableDegree[j];
            float temp       = -_LogLikelihoodRatio[j];
            for(int k = 0; k < degVi; k++) {
                temp += (zReplica[ t_row[ptr] ] + Lambda[ t_row[ptr] ]);
                ptr  +=1;
            }
            float xx = (temp -  tableau[degVi]) * tabCoef[degVi];///(deg - (2.0f * _amu_));
            OutputFromDecoder[j] = admm::max(admm::min(xx, 1.0f), 0.0f);
        }

        // CN processing kernel
        int CumSumCheckDegree = 0; // cumulative position of currect edge in factor graph
        int allVerified       = 0;
        float v_before_proj[16] __attribute__((aligned(64)));
        for(int j = 0; j < mNChecks; j++)
        {
            #if SKIP_PROJECTION == true
              float difference = 0.0f;
            #endif
            int syndrom = 0;
            #pragma unroll
            for(int k = 0; k < CheckDegree[j]; k++) // fill out the vector
            {
                const auto ind         = CumSumCheckDegree + k;
                const auto xpred       = OutputFromDecoder[ t_col1[ind] ];
                syndrom               += (xpred > 0.5);
                v_before_proj[k]  = rho * xpred + un_m_rho * zReplica[ind] - Lambda[ind];
#if SKIP_PROJECTION == true
				difference            += admm::abs(latestProjVector[ind] - v_before_proj[k]);
#endif
            }

            allVerified   += ( syndrom & 0x01 );

#if SKIP_PROJECTION == true
			if( difference > 1.0e-4 ){
#else
      if( true ) { //difference >= 0.0f ){
#endif

                //float* ztemp  = mp.mProjectionPPd(v_before_proj, CheckDegree[j]);
                float* ztemp  = mp.ProjectionInitiale(v_before_proj, CheckDegree[j]);
                #pragma unroll
                for(int k = 0; k < CheckDegree[j]; k++) {
                    const auto l        = CumSumCheckDegree + k;
                    Lambda[l]           = Lambda[l] + (rho * (ztemp[k] - OutputFromDecoder[ t_col1[l] ]) + un_m_rho * (ztemp[k] - zReplica[l]));
                    zReplica[l]         = ztemp[k];
#if SKIP_PROJECTION == true
                    latestProjVector[l] = v_before_proj[k];
#endif
                }
            }
            CumSumCheckDegree += CheckDegree[j];
        }
        if(allVerified == 0) {
            mAlgorithmConverge = true;
            mValidCodeword     = true;
            break;
        }
    }
}
#undef SKIP_PROJECTION
