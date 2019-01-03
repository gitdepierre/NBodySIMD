#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>

#include "math.h"
#include <time.h>

#if defined SSE||defined AVX||defined AVX512
#include "immintrin.h"
#endif
#ifdef NEON
#include "arm_neon.h"
#endif


#define NBody 2048

typedef struct BodyPos BodyPos;
struct BodyPos
{
    float posx, posy;
};

typedef struct BodyV BodyV;
struct BodyV
{
    float vx, vy;
};

float* tabBodyPosX;
float* tabBodyPosY;
float* tabBodyVX;
float* tabBodyVY;
float* tempTabX;
float* tempTabY;

const float invNBody=1.0f / NBody;

#ifdef AVX512
void Update()
{
    const __m512 decalage= _mm512_set1_ps(0.00001f);
    const __m512 inversion= _mm512_set1_ps(1.0f);
    const __m512 invNbodyAVX= _mm512_set1_ps(invNBody);

    #pragma omp parallel for schedule (dynamic,16)
    for (int i = 0; i<NBody; i++)
    {
        __m512 tabPosXi = _mm512_set1_ps(tabBodyPosX[i]);
        __m512 tabPosYi = _mm512_set1_ps(tabBodyPosY[i]);

        __m512 resultatAVXX=_mm512_setzero_ps();
        __m512 resultatAVXXTemp=_mm512_setzero_ps();
        __m512 resultatAVXY=_mm512_setzero_ps();
        __m512 resultatAVXYTemp=_mm512_setzero_ps();

        for (int j = 0; j<NBody; j +=32)
        {

            //Premier pack
            __m512 tabPosXj = _mm512_load_ps(&tabBodyPosX[j]);
            __m512 tabPosYj = _mm512_load_ps(&tabBodyPosY[j]);

            __m512 DiffX = _mm512_sub_ps(tabPosXi, tabPosXj);
            __m512 DiffY = _mm512_sub_ps(tabPosYi, tabPosYj);

            __m512 DiffXCarre = _mm512_mul_ps(DiffX, DiffX);
            __m512 DiffYCarre = _mm512_mul_ps(DiffY, DiffY);

            __m512 distanceCarre = _mm512_add_ps (DiffXCarre, DiffYCarre);
            distanceCarre = _mm512_add_ps (decalage, distanceCarre);

            __m512 DeltaX = _mm512_mul_ps (invNbodyAVX, _mm512_div_ps (DiffX, distanceCarre));
            __m512 DeltaY = _mm512_mul_ps (invNbodyAVX, _mm512_div_ps (DiffY, distanceCarre));

            resultatAVXXTemp = _mm512_add_ps(resultatAVXX, DeltaX);
            resultatAVXYTemp = _mm512_add_ps(resultatAVXY, DeltaY);

            tabPosXj = _mm512_load_ps(&tabBodyPosX[j+16]);
            tabPosYj = _mm512_load_ps(&tabBodyPosY[j+16]);

            DiffX = _mm512_sub_ps(tabPosXi, tabPosXj);
            DiffY = _mm512_sub_ps(tabPosYi, tabPosYj);

            DiffXCarre = _mm512_mul_ps(DiffX, DiffX);
            DiffYCarre = _mm512_mul_ps(DiffY, DiffY);

            distanceCarre = _mm512_add_ps (DiffXCarre, DiffYCarre);
            distanceCarre = _mm512_add_ps (decalage, distanceCarre);

            DeltaX = _mm512_mul_ps (invNbodyAVX, _mm512_div_ps (DiffX, distanceCarre));
            DeltaY = _mm512_mul_ps (invNbodyAVX, _mm512_div_ps (DiffY, distanceCarre));

            resultatAVXX = _mm512_add_ps(resultatAVXXTemp, DeltaX);
            resultatAVXY = _mm512_add_ps(resultatAVXYTemp, DeltaY);

        }

        tabBodyVX[i]-= _mm512_reduce_add_ps  (resultatAVXX);
        tabBodyVY[i]-= _mm512_reduce_add_ps  (resultatAVXY);
    }

    for (int i = 0; i<NBody; i++)
    {
        tempTabX[i]=tabBodyPosX[i]+tabBodyVX[i];
        tempTabY[i]=tabBodyPosY[i]+tabBodyVY[i];
    }
    memcpy(tabBodyPosX,tempTabX,NBody*sizeof(float));
    memcpy(tabBodyPosY,tempTabY,NBody*sizeof(float));
}

#endif


#ifdef AVX
void Update()
{
    const __m256 decalage= _mm256_set1_ps(0.00001f);
    const __m256 inversion= _mm256_set1_ps(1.0f);
    const __m256 invNbodyAVX= _mm256_set1_ps(invNBody);


    #pragma omp parallel for schedule (dynamic,16)
    for (int i = 0; i<NBody; i++)
    {
        const __m256 tabPosXi = _mm256_set1_ps(tabBodyPosX[i]);
        const __m256 tabPosYi = _mm256_set1_ps(tabBodyPosY[i]);
        __m256 resultatAVXX=_mm256_setzero_ps();
        __m256 resultatAVXXTemp=_mm256_setzero_ps();
        __m256 resultatAVXY=_mm256_setzero_ps();
        __m256 resultatAVXYTemp=_mm256_setzero_ps();


        for (int j = 0; j<NBody; j +=16)
        {


            //Premier pack
            __m256 tabPosXj = _mm256_load_ps(&tabBodyPosX[j]);
            __m256 tabPosYj = _mm256_load_ps(&tabBodyPosY[j]);

            __m256 DiffX = _mm256_sub_ps(tabPosXi, tabPosXj);
            __m256 DiffY = _mm256_sub_ps(tabPosYi, tabPosYj);

            __m256 DiffXCarre = _mm256_mul_ps(DiffX, DiffX);
            __m256 DiffYCarre = _mm256_mul_ps(DiffY, DiffY);

            __m256 distanceCarre = _mm256_add_ps (DiffXCarre, DiffYCarre);
            distanceCarre = _mm256_add_ps (decalage, distanceCarre);

            __m256 DeltaX = _mm256_mul_ps (invNbodyAVX, _mm256_div_ps (DiffX, distanceCarre));
            __m256 DeltaY = _mm256_mul_ps (invNbodyAVX, _mm256_div_ps (DiffY, distanceCarre));

            resultatAVXXTemp = _mm256_add_ps(resultatAVXX, DeltaX);
            resultatAVXYTemp = _mm256_add_ps(resultatAVXY, DeltaY);


            tabPosXj = _mm256_load_ps(&tabBodyPosX[j+8]);
            tabPosYj = _mm256_load_ps(&tabBodyPosY[j+8]);

            DiffX = _mm256_sub_ps(tabPosXi, tabPosXj);
            DiffY = _mm256_sub_ps(tabPosYi, tabPosYj);

            DiffXCarre = _mm256_mul_ps(DiffX, DiffX);
            DiffYCarre = _mm256_mul_ps(DiffY, DiffY);

            distanceCarre = _mm256_add_ps (DiffXCarre, DiffYCarre);
            distanceCarre = _mm256_add_ps (decalage, distanceCarre);

            DeltaX = _mm256_mul_ps (invNbodyAVX, _mm256_div_ps (DiffX, distanceCarre));
            DeltaY = _mm256_mul_ps (invNbodyAVX, _mm256_div_ps (DiffY, distanceCarre));

            resultatAVXX = _mm256_add_ps(resultatAVXXTemp, DeltaX);
            resultatAVXY = _mm256_add_ps(resultatAVXYTemp, DeltaY);


        }


        __m128 resultatAVXX1=_mm256_extractf128_ps(resultatAVXX,0);
        __m128 resultatAVXX2=_mm256_extractf128_ps(resultatAVXX,1);
        __m128 resultatAVXY1=_mm256_extractf128_ps(resultatAVXY,0);
        __m128 resultatAVXY2=_mm256_extractf128_ps(resultatAVXY,1);

        __m128 resultatAVXX128= _mm_hadd_ps(resultatAVXX1,resultatAVXX2);
        __m128 resultatAVXY128= _mm_hadd_ps(resultatAVXY1,resultatAVXY2);

        float* resultatX = (float*) &resultatAVXX128;
        float* resultatY = (float*) &resultatAVXY128;

        tabBodyVX[i]-= (resultatX[0]+resultatX[1]+resultatX[2]+resultatX[3]);
        tabBodyVY[i]-= (resultatY[0]+resultatY[1]+resultatY[2]+resultatY[3]);
    }

    for (int i = 0; i<NBody; i++)
    {
        tempTabX[i]=tabBodyPosX[i]+tabBodyVX[i];
        tempTabY[i]=tabBodyPosY[i]+tabBodyVY[i];
    }
    memcpy(tabBodyPosX,tempTabX,NBody*sizeof(float));
    memcpy(tabBodyPosY,tempTabY,NBody*sizeof(float));
}
#endif


#ifdef NEON
void Update()
{
    const float32x4_t decalage= vdupq_n_f32(0.00001f);
    const float32x4_t inversion= vdupq_n_f32(1.0f);
    const float32x4_t invNbodySSE= vdupq_n_f32(invNBody);

    #pragma omp parallel for schedule (dynamic,32)
    for (int i = 0; i<NBody; i++)
    {
        const float32x4_t tabPosXi = vdupq_n_f32(tabBodyPosX[i]);
        const float32x4_t tabPosYi = vdupq_n_f32(tabBodyPosY[i]);

        float32x4_t resultatSSEX=vdupq_n_f32(0);
        float32x4_t resultatSSEXTemp=vdupq_n_f32(0);
        float32x4_t resultatSSEY=vdupq_n_f32(0);
        float32x4_t resultatSSEYTemp=vdupq_n_f32(0);

        for (int j = 0; j<NBody; j +=8)
        {

            //Premier pack
            float32x4_t tabPosXj = vld1q_f32 (&tabBodyPosX[j]);
            float32x4_t tabPosYj = vld1q_f32 (&tabBodyPosY[j]);

            float32x4_t DiffX = vsubq_f32(tabPosXi, tabPosXj);
            float32x4_t DiffY = vsubq_f32(tabPosYi, tabPosYj);

            float32x4_t DiffXCarre = vmulq_f32 (DiffX, DiffX);
            float32x4_t DiffYCarre = vmulq_f32 (DiffY, DiffY);

            float32x4_t distanceCarre = vaddq_f32 (DiffXCarre, DiffYCarre);
            distanceCarre = vaddq_f32 (decalage, distanceCarre);

            float32x4_t DeltaX = vmulq_f32  (invNbodySSE, vmulq_f32 (DiffX, vrecpeq_f32(distanceCarre)));
            float32x4_t DeltaY = vmulq_f32  (invNbodySSE, vmulq_f32 (DiffY, vrecpeq_f32(distanceCarre)));

            resultatSSEXTemp = vaddq_f32(resultatSSEX, DeltaX);
            resultatSSEYTemp = vaddq_f32(resultatSSEY, DeltaY);

            tabPosXj = vld1q_f32 (&tabBodyPosX[j+4]);
            tabPosYj = vld1q_f32 (&tabBodyPosY[j+4]);

            DiffX = vsubq_f32(tabPosXi, tabPosXj);
            DiffY = vsubq_f32(tabPosYi, tabPosYj);

            DiffXCarre = vmulq_f32 (DiffX, DiffX);
            DiffYCarre = vmulq_f32 (DiffY, DiffY);

            distanceCarre = vaddq_f32 (DiffXCarre, DiffYCarre);
            distanceCarre = vaddq_f32 (decalage, distanceCarre);

            DeltaX = vmulq_f32  (invNbodySSE, vmulq_f32 (DiffX, vrecpeq_f32(distanceCarre)));
            DeltaY = vmulq_f32  (invNbodySSE, vmulq_f32 (DiffY, vrecpeq_f32(distanceCarre)));

            resultatSSEX = vaddq_f32(resultatSSEXTemp, DeltaX);
            resultatSSEY = vaddq_f32(resultatSSEYTemp, DeltaY);


        }


        float* resultatX = (float*) &resultatSSEX;
        float* resultatY = (float*) &resultatSSEY;

        tabBodyVX[i]-= (resultatX[0]+resultatX[1]+resultatX[2]+resultatX[3]);
        tabBodyVY[i]-= (resultatY[0]+resultatY[1]+resultatY[2]+resultatY[3]);
    }

    for (int i = 0; i<NBody; i++)
    {
        tempTabX[i]=tabBodyPosX[i]+tabBodyVX[i];
        tempTabY[i]=tabBodyPosY[i]+tabBodyVY[i];
    }
    memcpy(tabBodyPosX,tempTabX,NBody*sizeof(float));
    memcpy(tabBodyPosY,tempTabY,NBody*sizeof(float));
}
#endif

#ifdef SSE
void Update()
{
    const __m128 decalage= _mm_set1_ps(0.00001f);
    const __m128 inversion= _mm_set1_ps(1.0f);
    const __m128 invNbodySSE= _mm_set1_ps(invNBody);

    #pragma omp parallel for schedule (dynamic,32)
    for (int i = 0; i<NBody; i++)
    {
        const __m128 tabPosXi = _mm_set1_ps(tabBodyPosX[i]);
        const __m128 tabPosYi = _mm_set1_ps(tabBodyPosY[i]);

        __m128 resultatSSEX=_mm_setzero_ps();
        __m128 resultatSSEXTemp=_mm_setzero_ps();
        __m128 resultatSSEY=_mm_setzero_ps();
        __m128 resultatSSEYTemp=_mm_setzero_ps();

        for (int j = 0; j<NBody; j +=8)
        {

            //Premier pack
            __m128 tabPosXj = _mm_load_ps(&tabBodyPosX[j]);
            __m128 tabPosYj = _mm_load_ps(&tabBodyPosY[j]);

            __m128 DiffX = _mm_sub_ps(tabPosXi, tabPosXj);
            __m128 DiffY = _mm_sub_ps(tabPosYi, tabPosYj);

            __m128 DiffXCarre = _mm_mul_ps(DiffX, DiffX);
            __m128 DiffYCarre = _mm_mul_ps(DiffY, DiffY);

            __m128 distanceCarre = _mm_add_ps (DiffXCarre, DiffYCarre);
            distanceCarre = _mm_add_ps (decalage, distanceCarre);

            __m128 DeltaX = _mm_mul_ps (invNbodySSE, _mm_div_ps (DiffX, distanceCarre));
            __m128 DeltaY = _mm_mul_ps (invNbodySSE, _mm_div_ps (DiffY, distanceCarre));

            resultatSSEXTemp = _mm_add_ps(resultatSSEX, DeltaX);
            resultatSSEYTemp = _mm_add_ps(resultatSSEY, DeltaY);

            tabPosXj = _mm_load_ps(&tabBodyPosX[j+4]);
            tabPosYj = _mm_load_ps(&tabBodyPosY[j+4]);

            DiffX = _mm_sub_ps(tabPosXi, tabPosXj);
            DiffY = _mm_sub_ps(tabPosYi, tabPosYj);

            DiffXCarre = _mm_mul_ps(DiffX, DiffX);
            DiffYCarre = _mm_mul_ps(DiffY, DiffY);

            distanceCarre = _mm_add_ps (DiffXCarre, DiffYCarre);
            distanceCarre = _mm_add_ps (decalage, distanceCarre);

            DeltaX = _mm_mul_ps (invNbodySSE, _mm_div_ps (DiffX, distanceCarre));
            DeltaY = _mm_mul_ps (invNbodySSE, _mm_div_ps (DiffY, distanceCarre));

            resultatSSEX = _mm_add_ps(resultatSSEXTemp, DeltaX);
            resultatSSEY = _mm_add_ps(resultatSSEYTemp, DeltaY);

        }


        float* resultatX = (float*) &resultatSSEX;
        float* resultatY = (float*) &resultatSSEY;

        tabBodyVX[i]-= (resultatX[0]+resultatX[1]+resultatX[2]+resultatX[3]);
        tabBodyVY[i]-= (resultatY[0]+resultatY[1]+resultatY[2]+resultatY[3]);
    }

    for (int i = 0; i<NBody; i++)
    {
        tempTabX[i]=tabBodyPosX[i]+tabBodyVX[i];
        tempTabY[i]=tabBodyPosY[i]+tabBodyVY[i];
    }
    memcpy(tabBodyPosX,tempTabX,NBody*sizeof(float));
    memcpy(tabBodyPosY,tempTabY,NBody*sizeof(float));
}
#endif

#ifdef NONE
void Update()
{
    #pragma omp parallel for schedule (dynamic,32)
    for (int i = 0; i<NBody; i++)
    {

        float distanceCarre;
        float tempDistX;
        float tempDistY;

        for (int j = 0; j<NBody; j++)
        {
            tempDistX=tabBodyPosX[i]-tabBodyPosX[j];
            tempDistY=tabBodyPosY[i]-tabBodyPosY[j];
            distanceCarre=1.0f/((tempDistX*tempDistX)+(tempDistY*tempDistY)+0.00001f);
            tabBodyVX[i]-=((invNBody)*tempDistX*distanceCarre);
            tabBodyVY[i]-=((invNBody)*tempDistY*distanceCarre );
        }
    }

    for (int i = 0; i<NBody; i++)
    {
        tempTabX[i]=tabBodyPosX[i]+tabBodyVX[i];
        tempTabY[i]=tabBodyPosY[i]+tabBodyVY[i];
    }
    memcpy(tabBodyPosX,tempTabX,NBody*sizeof(float));
    memcpy(tabBodyPosY,tempTabY,NBody*sizeof(float));
}
#endif NONE

int main()
{

	#if defined LINUX || defined NEON
	tabBodyPosX = (float*)aligned_alloc(32,NBody* sizeof(float));
    tabBodyPosY = (float*)aligned_alloc(32,NBody* sizeof(float));
    tabBodyVX = (float*)aligned_alloc(32,NBody* sizeof(float));
    tabBodyVY = (float*)aligned_alloc(32,NBody* sizeof(float));
	#else
    tabBodyPosX = (float*)_aligned_malloc(NBody* sizeof(float),32);
    tabBodyPosY = (float*)_aligned_malloc(NBody* sizeof(float),32);
    tabBodyVX = (float*)_aligned_malloc(NBody* sizeof(float),32);
    tabBodyVY = (float*)_aligned_malloc(NBody* sizeof(float),32);
	#endif
	
    tempTabX = (float*)calloc(NBody, sizeof(float));
    tempTabY = (float*)calloc(NBody, sizeof(float));

    srand(374); //Deterministic random
	
    for (int i = 0; i < NBody; i++)
    {
        tabBodyPosX[i] = 600 * (-0.5f + (rand() /(float) RAND_MAX));
        tabBodyPosY[i] = 300 * (-0.5f + (rand() / (float)RAND_MAX));
    }

    int compteur=0;
    struct timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);
    struct timespec specOld=spec;

    while (1)
    {
        Update();
		compteur++;
		if (compteur == 60)
		{
            clock_gettime(CLOCK_REALTIME, &spec);

            printf("time for 60 frames: %f\n",(spec.tv_sec-specOld.tv_sec)+(spec.tv_nsec-specOld.tv_nsec)/1000000000.0f );
			compteur = 0;
			specOld=spec;
		}
    }
    return 0;
}

