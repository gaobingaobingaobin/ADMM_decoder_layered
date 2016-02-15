/*
 *
 *  Created on: 17 avr. 2015
 *      Author: legal
 *
 * stable version: june 24th, 2015
 * still under modification
 */
#include <x86intrin.h>
#include <xmmintrin.h>
#include <chrono>
#include "./simdlib/admm.hpp"

#ifndef DECODER_HPP_
#define DECODER_HPP_

//#define PROFILE_ON

#include "MatriceProjection.hpp"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;


#define _type_ float

class Decoder{
public:
	//_type_* vector_before_proj;
	#define NEW_PROJECTION
	MatriceProjection<_type_, 32> mp;

	int get_deg_vn(int pos){
		return VariableDegree[pos];
	}

private:

	string DecoderName;
	string ChannelName;
	string mRealDecoderName;

	double* mProjectionPPd   (NODE v[], int length);
	double* mProjectionPPd_d6(NODE v[]);
	double* mProjectionPPd_d7(NODE v[]);

    long long int t_vn = 0;
    long long int t_cn = 0;
    long long int t_pj = 0;
    long long int t_ex = 0;

	bool hlayered;
	bool vlayered;

	// parameters for decoders
	string mPCMatrixFileName;  /*!< parity check matrix file */
	int mNChecks, mBlocklength, mPCheckMapSize;
	int *CheckDegree;
	int *VariableDegree;
	double RateOfCode; /*!< code rate, k/n */

	TRIPLE *mPCheckMap; /*!< parity check mapping */
	TRIPLE *mParityCheckMatrix; /*!< parity check matrix */

	unsigned int* t_col;
	unsigned int* t_row;

	unsigned int* t_col1;
	unsigned int* t_row1;

	unsigned int* t_row_stride_8;
	unsigned int* t_col_stride_8;

	float *Lambda;
	float *zReplica;
	float *latestProjVector;

	// algorithm parameter
	double para_end_feas;   /*!< ADMM alg end tolerance, typically 1e-5 */
	double para_mu;         /*!< ADMM alg \mu, typically 5.5 */
	double para_rho;        /*!< ADMM over relaxation para, typically 1.8, 1.9 */
	int maxIteration;       /*!< max iteration */

	//! Learn the degree of the parity check matrix
	/*!
	  The parity check matrix should be in correct format. This function is useful for irregular LDPC codes.
	*/
	void mLearnParityCheckMatrix();
	//! Read and store the parity check matrix
	void mReadParityCheckMatrix();
	void mSetDefaultParameters();


	//! Calculate log likelihood ratio using the output from the channel
	/*!
	  \param channeloutput Output from channel.
	  \sa NoisySeq
	*/
	void mGenerateLLR(NoisySeq &channeloutput);

	//Decoders:
	void ADMMDecoder_float_l2( );
	void ADMMDecoder_hlayered();
	void ADMMDecoder_vlayered();


	// data
	float* _LogLikelihoodRatio; /*!< log-likelihood ratio from received vector */
	float* OutputFromDecoder; /*!<soft information from decoder. could be pseudocodeword */

	double alpha;       /*!< the constant for penalty term */

	float  o_value[32]; /*!< the constant for penalty term */

	//decoder stats
	unsigned long int mExeTime; /*!< exevution time */
	bool mAlgorithmConverge; /*!< true if the decoder converges*/
	bool mValidCodeword; /*!< true if the output is a valid codeword */
	int mIteration; /*!< number of iterations used for decoding */

public:
	//! Set decoder parameters.
	/*!
	 \param mxIt Maximum number of iterations. Default: 1000
	 \param feas ADMM stopping parameter. Default: 1e-6
	 \param p_mu ADMM step parameter mu. Default: 5
	 \param p_rho ADMM over relaxation parameter. Default: 1.8
	*/
	void SetParameters(int mxIt, double feas, double p_mu, double p_rho);
	void SetDecoder(string decoder);
	void SetChannel(string channel){ChannelName = channel;}
	void SetPenaltyConstant(double alp){alpha = alp;}
	bool ValidateCodeword();
	//! See if the input sequence is valid codeword
	/*!
		\param input Input sequence array.
	*/
	bool ValidateCodeword(int input[]);
	//! Invoke decode function
	/*!
		\param channeloutput Noisy sequence from channel.
		\param decoderesult Decode results. Contains decoded sequence and other information.
		\sa DecodedSeq
		\sa NoisySeq
	*/
	double Decode(NoisySeq &channeloutput, DecodedSeq &decoderesult);

	//! Constructor.
	/*!
		\param FileName File for parity check matrix.
		\param blocklength Blocklength of code (i.e. n)
		\param nChecks Number of checks (i.e. row number of parity check matrix or (n-k))
	*/
	Decoder(string FileName,int nChecks, int BlockLength);
	//! Destructor
	~Decoder();

	static bool first_run;

	///////////////////////////////////////////////////////////////////////////////////////////////

	void setHorizontalLayeredDecoding( bool enable )
	{
        hlayered = enable;
        vlayered = false;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	void setVerticalLayeredDecoding( bool enable )
	{
        vlayered = enable;
        hlayered = false;

        //
        // LE CONTENU DU TABLEAU TABLE T_ROW1 DOIT ETRE REMPLACE
        //
        if( vlayered == true )
        {
            int ptr = 0;
            for(int i = 0; i < mBlocklength; i++) {
                for(int k = 0; k < mPCheckMapSize; k++) {
                    if( mPCheckMap[k].col == i ) {
                        t_row1[ptr] = mParityCheckMatrix[k].row;
                        ptr += 1;
                    }
                }
            }
            
            
            int CumSumVarDegree = 0;
            for (int i=0; i < mBlocklength ; i++) {
                for (int j = 0 ; j < VariableDegree[i] ; j++) {
                    const auto  ind = CumSumVarDegree + j;
                    int ind_start=0;
                    for (int count=0; count <  t_row1[ind]; count++)
                    {
                        ind_start += CheckDegree[ count ];
                    }
                    t_row_stride_8[CumSumVarDegree + j] = ind_start;
                }
                CumSumVarDegree += VariableDegree[i];
            }
        }
    }
	///////////////////////////////////////////////////////////////////////////////////////////////

};


//#define STORE_IN_FILE

bool Decoder::first_run = true;

// Decoder Class
Decoder::Decoder(string FileName,int nChecks, int BlockLength) : mp()
{
#ifdef PROFILE_ON
    t_vn = 0;
    t_cn = 0;
    t_pj = 0;
    t_ex = 0;
#endif

	hlayered    = false; // BLG
	vlayered    = false;

	// Learn and read parity check matrix
	mPCMatrixFileName = FileName;
	mNChecks = nChecks;
	mBlocklength = BlockLength;
	mPCheckMapSize = 0;
	CheckDegree         = (int*)_mm_malloc(mNChecks * sizeof(int), 64);
	VariableDegree      = (int*)_mm_malloc(mBlocklength * sizeof(int), 64);
	OutputFromDecoder   = (float*)_mm_malloc(mBlocklength * sizeof(float), 64);
	_LogLikelihoodRatio = (float*)_mm_malloc(mBlocklength * sizeof(float), 64);

	mSetDefaultParameters();
	mLearnParityCheckMatrix();
	mPCheckMap = new TRIPLE[mPCheckMapSize];
	mParityCheckMatrix = new TRIPLE[mPCheckMapSize];
	mReadParityCheckMatrix();
	RateOfCode = (double)(mBlocklength -mNChecks)/mBlocklength;
	//vector_before_proj = (float*)_mm_malloc(64 * sizeof(float), 64);

	Lambda   = (float*)_mm_malloc(mPCheckMapSize * sizeof(float), 64);
	zReplica = (float*)_mm_malloc(mPCheckMapSize * sizeof(float), 64);
	latestProjVector = (float*)_mm_malloc(mPCheckMapSize * sizeof(float), 64);

	int ptr = 0;
	t_col   = (unsigned int*)_mm_malloc(mPCheckMapSize * sizeof(unsigned int), 64);
	for(int k = 0; k < mNChecks; k++)
	{
		for(int i = 0; i < mPCheckMapSize; i++)
		{
			if( mParityCheckMatrix[i].row == k )
			{
				t_col[ptr] = mParityCheckMatrix[i].col;
				ptr += 1;
			}

		}
	}

#ifdef STORE_IN_FILE
	FILE* f1 = fopen("t_col.txt", "w");
	if( f1 != NULL )
	{
		fprintf(f1, "int t_col[%d] = {\n", mPCheckMapSize);
		for(int p=0; p<mPCheckMapSize; p++)
		{
			if( p%CheckDegree[0] == 0 ) fprintf(f1, "\n");
			fprintf(f1, "%4d, ", t_col[p]);
		}
		fprintf(f1, "\n");
		fclose ( f1 );
	}
#endif

    ptr    = 0;
	t_col1 = (unsigned int*)_mm_malloc(mPCheckMapSize * sizeof(unsigned int), 64);
	for(int k = 0; k < mPCheckMapSize; k++)
	{
		for(int i = 0; i < mPCheckMapSize; i++)
		{
			if( mPCheckMap[i].row == k )
			{
				t_col1[ptr] = mPCheckMap[i].col;
				ptr += 1;
                                break; // on en a fini avec le msg_k
			}

		}
	}

#ifdef STORE_IN_FILE
	FILE* f2 = fopen("t_col1.txt", "w");
	if( f2 != NULL )
	{
		fprintf(f2, "int t_col[%d] = {\n", mPCheckMapSize);
		for(int p=0; p<mPCheckMapSize; p++)
		{
			if( p%CheckDegree[0] == 0 ) fprintf(f2, "\n");
			fprintf(f2, "%4d, ", t_col[p]);
		}
		fprintf(f2, "\n");
		fclose ( f2 );
	}
#endif

    ptr   = 0;
	t_row = (unsigned int*)_mm_malloc(mPCheckMapSize * sizeof(unsigned int), 64);
	for(int j = 0; j < mBlocklength; j++)
	{
		for(int i = 0; i < mPCheckMapSize; i++)
		{
			if( mPCheckMap[i].col == j )
			{
				t_row[ptr] = mPCheckMap[i].row;
				ptr += 1;
			}
		}
	}

#ifdef STORE_IN_FILE
	FILE* f3 = fopen("t_row.txt", "w");
	if( f3 != NULL )
	{
		fprintf(f3, "int t_row[%d] = {\n", mPCheckMapSize);
		for(int p=0; p<mPCheckMapSize; p++)
		{
			if( p%VariableDegree[0] == 0 ) fprintf(f3, "\n");
			fprintf(f3, "%4d, ", t_row[p]);
		}
		fprintf(f3, "\n\n");
		fprintf(f3, "short t_row_pad_4[%d] = {\n", (4*mPCheckMapSize)/3);
		for(int p=0; p<mPCheckMapSize; p++)
		{
			if( p%VariableDegree[0] == 0 ) fprintf(f3, " 0 /* PADDING */,\n");
			fprintf(f3, "%4d, ", t_row[p]);
		}
		fprintf(f3, " 0};\n");
		fclose ( f3 );
		exit( 0 );
	}else{
		printf("(EE) Error opening file : t_row.txt\n");
		exit( 0 );
	}
#endif

    ptr     = 0;
	t_row1  = (unsigned int*)_mm_malloc(mPCheckMapSize * sizeof(unsigned int), 64);
	for(int j = 0; j < mBlocklength; j++)
	{
		for(int i = 0; i < mPCheckMapSize; i++)
		{
			if( mPCheckMap[i].col == j )
			{
				t_row1[ptr] = mPCheckMap[i].row;
 				ptr += 1;
			}
		}
	}

    int offset = 0;
#ifdef STORE_IN_FILE
	ofstream ft_row("t_row_original.txt");
	for(int j = 0; j < mBlocklength; j+=1)
	{
		ft_row << "VN " << j << " : ";
		int degVn = VariableDegree[j];
		for(int k = 0; k < degVn; k++)
		{
			ft_row << t_row[offset + k] << j << ", ";
		}
		ft_row << endl;
		offset += degVn;
	}
	exit( 0 );
#endif

    ptr            = 0;
	t_row_stride_8 = (unsigned int*)_mm_malloc(mPCheckMapSize * sizeof(unsigned int), 64);
    offset = 0;
    bool failed = false;
	for(int j = 0; j < mBlocklength; j+=8)
	{
		int degVn = VariableDegree[j];
		for(int k = 0; k < 8; k++)
		{
			if( degVn != VariableDegree[j+k] ){

				if( first_run == true )printf("(EE) Error in node reodering (t_row_stride_8) !\n");
				if( first_run == true )printf("(EE) - Initial VN : %d\n", degVn);
				if( first_run == true )printf("(EE) - Current VN : %d\n", VariableDegree[j+k]);
				//for(int qq = 0; qq < mBlocklength; qq+=1)
				//{
				//	if( qq % 32 == 0 ) printf("\n");
				//	if( qq %  8 == 0 ) printf("  |  ");
				//	printf("%2d ", VariableDegree[qq]);
				//}printf("\n");
				failed = true;
				break;
			}
			if( failed == true )
			{
				break;
			}
			for(int i = 0; i < VariableDegree[j+k]; i++)
			{
				t_row_stride_8[offset + 8 * i + k] = t_row[offset + k * VariableDegree[j] + i];
			}
		}
		offset += 8 * VariableDegree[j];
	}


	//
	// ON ENTRELACE LA TABLE DE TRANSLATION DES MESSAGES AU NIVEAU DES CNs POUR PERMETTRE UN
	// TRAITEMENT EN LOT DES NOEUDS CNs
	//
	t_col_stride_8  = (unsigned int*)_mm_malloc(mPCheckMapSize * sizeof(unsigned int), 64);
	unsigned int* t = (unsigned int*)_mm_malloc(mPCheckMapSize * sizeof(unsigned int), 64);
    ptr    = 0;
    offset = 0;

    for(int j = 0; j < mPCheckMapSize; j+=8)
	{
    	t[j] = j;
	}
    bool failed2 = false;
    for(int j = 0; j < mNChecks; j+=8)
	{
		int degCn = CheckDegree[j];
		for(int k = 0; k < 8; k++)
		{
			if( degCn != CheckDegree[j+k] ){
				if( first_run == true )printf("(EE) Error in node reodering (t_col_stride_8) !\n");
				if( first_run == true )printf("(EE) - Initial CN : %d\n", degCn);
				if( first_run == true )printf("(EE) - Current CN : %d\n", CheckDegree[j+k]);
				//for(int qq = 0; qq < mNChecks; qq+=1)
				//{
				//	if( qq % 32 == 0 ) printf("\n");
				//	if( qq %  8 == 0 ) printf("  |  ");
				//	printf("%2d ", CheckDegree[qq]);
				//}printf("\n");
				failed2 = true;
				break;
			}
			if( failed2 == true )
			{
				break;
			}
			for(int i = 0; i < CheckDegree[j+k]; i++)
			{
				t_col_stride_8[offset + 8 * i + k] = t_col1[offset + k * CheckDegree[j] + i];
				t             [offset + 8 * i + k] =        offset + k * CheckDegree[j] + i;
			}
		}
		offset += 8 * CheckDegree[j];
	}

#if 0
    printf("Permutation translation table:\n");
	for( int j = 0; j < mPCheckMapSize; j+=1 )
	{
		if( j%16 == 0 ) printf("\n");
		printf("%4d, ", t[j]);
	}printf("\n\n");

    printf("Initial t_row (after row perm)\n");
	for( int j = 0; j < 128; j+=1 )
	{
		if( j%16 == 0 ) printf("\n");
		printf("%4d, ", t_row_stride_8[j]);
	}printf("\n\n");
#endif

#if 1
	for( int j = 0; j < mPCheckMapSize; j+=1 )
	{
		int prev_ptr = t_row_stride_8[j];
		for( int q = 0; q < mPCheckMapSize; q += 1 )
		{
			if( t[q] == prev_ptr ){
				t_row_stride_8[j] = q;
				break;
			}
		}
	}
#endif

#if 0
    printf("Final t_row (after CN perm)\n");
	for( int j = 0; j < mPCheckMapSize; j+=1 )
	{
		if( j%16 == 0 ) printf("\n");
		printf("%4d, ", t_row_stride_8[j]);
	}printf("\n\n");
	exit( 0 );
#endif
	_mm_free( t );

	first_run = false;
}


Decoder::~Decoder()
{
#ifdef PROFILE_ON
    if( t_ex != 0 ){
        t_cn /= t_ex;
        t_vn /= t_ex;
        t_pj /= t_ex;
        long long int sum = t_cn + t_vn + t_pj;
        double CN = (100.0 * t_cn) / (double)sum;
        double VN = (100.0 * t_vn) / (double)sum;
        double PJ = (100.0 * t_pj) / (double)sum;
        printf("ALL : %lld - VN : %lld - CN = %lld (PJ = %lld)\n", t_ex, t_cn, t_vn, t_pj);
        printf("VN : %1.3f - CN = %1.3f - PJ = %1.3f\n", VN, CN, PJ);
    }else{
    	printf("(WW) PROFILE WAS SWITWED ON AND NO DATA ARE AVAILABLE !\n");
    }
#endif

	_mm_free ( CheckDegree );
	_mm_free ( VariableDegree );
	delete [] mPCheckMap;
	delete [] mParityCheckMatrix;
	_mm_free ( OutputFromDecoder );
	_mm_free (_LogLikelihoodRatio );
	//_mm_free ( vector_before_proj );
	_mm_free ( t_col );
	_mm_free ( t_row );
	_mm_free ( t_col_stride_8 );
	_mm_free ( t_row_stride_8 );
	_mm_free ( t_col1 );
	_mm_free ( t_row1 );
	_mm_free ( Lambda );
	_mm_free ( zReplica );
	_mm_free( latestProjVector );
}

//Decodersettings
void Decoder::SetParameters(int mxIt, double feas,  double p_mu, double p_rho)
{
	maxIteration  = mxIt;
	para_end_feas = feas;
	para_mu       = p_mu;
	para_rho      = p_rho;
}



void Decoder::SetDecoder(string decoder)
{
	mRealDecoderName = decoder;
        DecoderName  = "ADMM";
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void Decoder::mSetDefaultParameters()
{
	// decoder settings
	maxIteration = 1000;
	para_end_feas = 1e-6;
	para_mu = 5;
	para_rho= 1.8;

	alpha = 1;
	for(int i = 0; i < mBlocklength; i++)
	{
		VariableDegree[i] = 0;
	}
	for(int i = 0; i < mNChecks; i++)
	{
		CheckDegree[i] = 0;
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void Decoder::mLearnParityCheckMatrix()// learn the degree of the parity check matrix
{
	ifstream myfile (mPCMatrixFileName.c_str());
	int tempvalue = 0;
	int i = 0;
	string line;
	if (myfile.is_open())
	{
		while(getline(myfile, line))
		{
		   istringstream is(line);
		   int curr_degree = 0;
		   while( is >> tempvalue )
		   {
			   VariableDegree[tempvalue]++;
			   curr_degree++;
			   mPCheckMapSize++;
		   }
		   CheckDegree[i] = curr_degree;
		   i++;
		}
		if (myfile.is_open())
			myfile.close();
	}
	else
	{
		cout << "Unable to open file";
		exit(0);
	}
//	cout <<"mPCheckMapSize= "<< mPCheckMapSize<<endl;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void Decoder::mReadParityCheckMatrix()// read the parity check matrix, initialize message passing
{
	ifstream myfile (mPCMatrixFileName.c_str());
	int tempvalue = 0;
	int i = 0;
	int count = 0;
	string line;
	if (myfile.is_open())
	{
		while(getline(myfile, line))
		{
			istringstream is(line);
			while( is >> tempvalue )
			{
				mParityCheckMatrix[count].col = tempvalue;
				mParityCheckMatrix[count].row = i;

				mPCheckMap[count].col = tempvalue;
				mPCheckMap[count].row = count;
				count++;
			}
			i++;
		}
	}
	if (myfile.is_open())
		myfile.close();
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void Decoder::mGenerateLLR(NoisySeq &channeloutput)
{
	if(ChannelName == "AWGN")
	{
		double std = channeloutput.ChannelParameter;
		for(int i = 0; i < mBlocklength; i++)
		{
			double receivedbit = channeloutput.NoisySequence[i];
			_LogLikelihoodRatio[i] = ((receivedbit - 1)*(receivedbit - 1) - (receivedbit + 1)*(receivedbit + 1)) / 2.0 / std / std;
		}
	}
	else if(ChannelName == "BSC")
	{
		double CrossOverProb = channeloutput.ChannelParameter;
		for(int i = 0; i < mBlocklength; i++)
		{
			int receivedbit = floor(channeloutput.NoisySequence[i] + 0.5);
			if(receivedbit == 0)
				_LogLikelihoodRatio[i] = - log(CrossOverProb/(1 - CrossOverProb));
			else
				_LogLikelihoodRatio[i] = log(CrossOverProb/(1 - CrossOverProb));
		}
	}
	else
	{
		cout<<"channel not supported"<<endl;
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



double Decoder::Decode(NoisySeq &channeloutput, DecodedSeq &decoderesult)
{
	mExeTime           = 0;
	mAlgorithmConverge = false;
	mValidCodeword     = false;

	mGenerateLLR(channeloutput);

	clock_t tStart = clock();
    auto start     = chrono::steady_clock::now();

    if( hlayered == true ){
        ADMMDecoder_hlayered( );

		}else if( vlayered == true ){
        ADMMDecoder_vlayered( );

    }else{
	ADMMDecoder_float_l2( );
    }

    auto end                       = chrono::steady_clock::now();
	mExeTime                       = clock() - tStart;
	decoderesult.ExeTime           = mExeTime;
	decoderesult.Iteration         = mIteration;
	decoderesult.ValidCodeword     = mValidCodeword;
	decoderesult.AlgorithmConverge = mAlgorithmConverge;
	decoderesult.Iteration         = mIteration;

	for(int i = 0; i < mBlocklength; i++)
	{
		if(my_isnan(OutputFromDecoder[i]))
			decoderesult.IsNan = true;
		if(my_isinf(OutputFromDecoder[i]))
			decoderesult.IsInf = true;
		decoderesult.SoftInfo[i]     = OutputFromDecoder[i];
		decoderesult.HardDecision[i] = floor(OutputFromDecoder[i] + 0.5);
	}
    auto diff      = end - start;
    return chrono::duration <double, milli> (diff).count();
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void show6(float* a)
{
	printf("__m256 : %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f\n", a[5], a[4], a[3], a[2], a[1], a[0]);
}

void show8(float* a)
{
	printf("__m256 : %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f\n", a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);
}

void show6(int* a)
{
	printf("__m256 : %d %d %d %d %d %d\n", a[5], a[4], a[3], a[2], a[1], a[0]);
}

void show8(int* a)
{
	printf("__m256 : %d %d %d %d %d %d %d %d\n", a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "decoders/decoder_float.hpp"
#include "decoders/decoder_hlayered.hpp"
#include "decoders/decoder_vlayered.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



bool Decoder::ValidateCodeword()
{
	int ptr = 0;
	for(int k = 0; k < mNChecks; k++)
	{
		int syndrom = 0;
		int deg     = CheckDegree[k];
		for(int i = 0; i < deg; i++)
		{
			syndrom += (int) floor(OutputFromDecoder[ t_col[ptr] ] + 0.5);
			ptr += 1;
		}
		if( syndrom & 0x01 ) return false;
	}
	return true;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



bool Decoder::ValidateCodeword(int input[])
{
	int ptr = 0;
	for(int k = 0; k < mNChecks; k++)
	{
		int syndrom = 0;
		int deg     = CheckDegree[k];
		for(int i = 0; i < deg; i++)
		{
			syndrom += (int) floor(input[ t_col[ptr] ]);
			ptr += 1;
		}
		if( syndrom & 0x01 ) return false;
	}
	return true;
}

#endif /* DECODER_HPP_ */
