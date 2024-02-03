/////////////////////readme/////////////////////////////////////////////////////////////////////////
//This is source code of using GPU CUDA to solve time-dependent Ginzburg Landau equations. The algorithm 
//paper is W. D. Gropp, et al., Journal of Computational Physics 123, 254 (1996). The developement environment
//is CUDA Toolkit 3.2.
//////////////////////////////////////////////////////////////////////////////////////////////////



#include "cuComplex.h"
#include "math.h"
#include <iostream>
#include <fstream>
using namespace std;
#define PRECISION	1.0e-6
#define Nx 89 //Lx/dx+1
#define Ny 45   //Ly/dy+1
#define kappa 2
#define sigma 1.0
#define BLOCK_SIZE 16
#define ApproachingSteps 2.0//The steps two holes approach to each other
#define ExpandSteps 4.0//broaden the holes

////////////////////global variable on Host//////////////////
    size_t sizeComplex; //memory address number for complex needed
    size_t sizeDouble;  //memory address number for double float
    int stride; //address number adder per row
	int width;  //grid width
	int height; //grid height

	double *temp;
	double hosttemp;
///////////////////give the parameter on device////////////////
    __device__ __constant__ double d_dt=0.00002; 
    __device__ __constant__ double d_Lx=20.0;
    __device__ __constant__ double d_Ly=8.0;
    __device__ __constant__ double d_dx=0.01;
    __device__ __constant__ double d_dy=0.01;
	__device__ __constant__ int tempLeftBoun1=int(Ny/4.0+ApproachingSteps+0.9-ExpandSteps);//set the coordinate of the boundary of holes
	__device__ __constant__ int tempRightBoun1=int(3.0*Ny/4.0+ApproachingSteps+0.01+ExpandSteps);
	__device__ __constant__ int tempLeftBoun2=int(Nx/2+Ny/4.0-ApproachingSteps+0.9-ExpandSteps);
	__device__ __constant__ int tempRightBoun2=int(Nx/2+3.0*Ny/4.0-ApproachingSteps+0.01+ExpandSteps);
	__device__ __constant__ int tempUpperBoun1=int(3.0*Ny/4.0+0.01+ExpandSteps);
	__device__ __constant__ int tempLowerBoun1=int(Ny/4.0+0.9-ExpandSteps);
	__device__ __constant__ int tempUpperBoun2=int(3.0*Ny/4.0+0.01+ExpandSteps);
	__device__ __constant__ int tempLowerBoun2=int(Ny/4.0+0.9-ExpandSteps);
	__device__ int LeftBoun1;
	__device__ int RightBoun1;
	__device__ int LeftBoun2;
	__device__ int RightBoun2;
	__device__ int UpperBoun1;
	__device__ int UpperBoun2;
	__device__ int LowerBoun1;
	__device__ int LowerBoun2;
//	__device__ double d_Ba;

//////////////////a class on device/////////////////
__device__ class CUDATDGL{
public:
	cuDoubleComplex d_Ux[Nx+1][Ny+1];
    cuDoubleComplex d_Uy[Nx+1][Ny+1];
    cuDoubleComplex d_W[Nx+1][Ny+1];
    cuDoubleComplex d_Psi[Nx+1][Ny+1];
    cuDoubleComplex d_dUxdt[Nx+1][Ny+1];
    cuDoubleComplex d_dUydt[Nx+1][Ny+1];
    cuDoubleComplex d_dPsidt[Nx+1][Ny+1];
	cuDoubleComplex tempPsi1[Nx+1][Ny+1];//used to store order parameters of upper and lower boundary when each corner point of the hole has two values
    double d_ImFux[Nx+1][Ny+1];
    double d_ImFuy[Nx+1][Ny+1];
    double d_x[Nx+1][Ny+1];
    double d_y[Nx+1][Ny+1];
    double d_Bz[Nx+1][Ny+1];
	double d_Jsx[Nx+1][Ny+1];
	double d_Jsy[Nx+1][Ny+1];
	double d_Ax[Nx+1][Ny+1];
	double d_Ay[Nx+1][Ny+1];
    double d_VOR[Nx+1][Ny+1]; //use dichotomy to calculate the number of vortices
	double d_ENG[Nx+1][Ny+1]; //use dichotomy to calculate the total energy
	int Totaltime;
    __device__ void CUinitMesh(int row,int col);
    __device__ void CUcalW(int row,int col,double *d_Ba);
    __device__ void CUcaldPsidt(int row,int col);
	__device__ void CUcaldUdt(int row, int col);
    __device__ void CUcalBC(int row,int col);
    __device__ void CUonestep(int row,int col);
	__device__ void CUcalJs(int row,int col);
	__device__ void CUcalNumOfVortices(int row,int col);
	__device__ void CUcalSysEng(int row,int col,double *d_Ba);
}CGL;

class HOSTTDGL{
public:
	cuDoubleComplex Psi[Nx+1][Ny+1];	//order parameter
	cuDoubleComplex Ux[Nx+1][Ny+1];
	cuDoubleComplex Uy[Nx+1][Ny+1];
	cuDoubleComplex W[Nx+1][Ny+1];
	double VOR[Nx+1][Ny+1];//VORcache in GPU
	double ENG[Nx+1][Ny+1];//ENGcache in GPU
	double Bz[Nx+1][Ny+1];
	double Jsx[Nx+1][Ny+1];
	double Jsy[Nx+1][Ny+1];
	double Ax[Nx+1][Ny+1];
	void ini_par();
	void CalTotal(double Ba);
	void OutputPsi(int fileid);
	void OutputJsx(int fileid);
	void OutputJsy(int fileid);
	void OutputPsiPhase(int fileid);
	void OutputBz(int fileid);
	void OutputAx(int fileid);
	void OutputUx(int fileid);
	double dt;
	double Lx;
	double Ly;
	double dx;
	double dy;
	double NumOfVortices;
	double SysEng;
	double averPsi;
	double magnetization;
	double nameBa;//for outputing file's name
	int count;//count how many points of magentization have been considered
}HGL;

void HOSTTDGL::ini_par(){
    dt=0.001;
	Lx=20.0;
    Ly=8.0;
	dx=0.1;
	dy=0.1;
}


__device__ void CUDATDGL::CUinitMesh(int row,int col){  //initialize mesh
	cuDoubleComplex tc1=make_cuDoubleComplex(0.0, 0.0);
	cuDoubleComplex tc2=make_cuDoubleComplex(cos(1.0),sin(1.0));
	d_Psi[col][row]=d_Ux[col][row]=d_Uy[col][row]=tc2;
	tempPsi1[col][row]=tc1;
	if(row>=tempLowerBoun1&&row<=tempUpperBoun1&&col>=tempLeftBoun1&&col<=tempRightBoun1){//build the two holes
	    d_Psi[col][row]=tc1;
	}
	if(row>=tempLowerBoun2&&row<=tempUpperBoun2&&col>=tempLeftBoun2&&col<=tempRightBoun2){
	    d_Psi[col][row]=tc1;
	}
	d_dPsidt[col][row]=d_dUxdt[col][row]=d_dUydt[col][row]=d_W[col][row]=tc1;
	d_ImFux[col][row]=0.0;
	d_ImFuy[col][row]=0.0;
	d_Jsx[col][row]=0.0;
	d_Jsy[col][row]=0.0;
	d_x[col][row]=col*d_dx;
	d_y[col][row]=row*d_dy;
	d_Bz[col][row]=0.0;
	d_VOR[col][row]=0.0;
    d_ENG[col][row]=0.0;
	d_Ax[col][row]=0.0;
	d_Ay[col][row]=0.0;
	if(col==Nx&&row==Ny){
	    Totaltime=0;
	    LeftBoun1=tempLeftBoun1;
	    RightBoun1=tempRightBoun1;
	    LeftBoun2=tempLeftBoun2;
	    RightBoun2=tempRightBoun2;
	    UpperBoun1=tempUpperBoun1;
	    UpperBoun2=tempUpperBoun2;
	    LowerBoun1=tempLowerBoun1;
	    LowerBoun2=tempLowerBoun2;
	}
}


__device__ void CUDATDGL::CUcalW(int row,int col,double *d_Ba){//Calculate W when U is known, Ba determines one boundary condition
    /*if(Totaltime<2000*100){
	    if(col<Nx&&row<Ny){
	        if(row==0||row==Ny-1||col==int(Nx/2+2)||col==0){
		        cuDoubleComplex tc1=make_cuDoubleComplex(1.0, -kappa*(*d_Ba)*d_dx*d_dy);
		        d_W[col][row]=tc1;
	        }
		    else{
	        d_W[col][row]= cuCmul(cuCmul(d_Ux[col][row],d_Uy[col+1][row]),\
                        cuCmul(cuConj(d_Ux[col][row+1]),cuConj(d_Uy[col][row])));
            }   
	    }
	}
	else{*/
	    if(col<Nx&&row<Ny){
	        if(row==0||row==Ny-1||col==0||col==Nx-1){
		        cuDoubleComplex tc1=make_cuDoubleComplex(1.0, -kappa*(*d_Ba)*d_dx*d_dy);
		        d_W[col][row]=tc1;
	        }
		    else{
	        d_W[col][row]= cuCmul(cuCmul(d_Ux[col][row],d_Uy[col+1][row]),\
                        cuCmul(cuConj(d_Ux[col][row+1]),cuConj(d_Uy[col][row])));
            }   
	    }
//	}
}



__device__ void CUDATDGL::CUcaldPsidt(int row,int col){
	tempPsi1[col][row]=d_Psi[col][row];
    if(row>0&&row<Ny&&col>0&&col<Nx&&(!(row>=LowerBoun1&&row<=UpperBoun1&&col>=LeftBoun1&&col<=RightBoun1))\
		&&(!(row>=LowerBoun2&&row<=UpperBoun2&&col>=LeftBoun2&&col<=RightBoun2))){
	    cuDoubleComplex tpsi1=cuCmul(d_Psi[col][row],cuConj(d_Psi[col][row]));
		cuDoubleComplex tpsi2=make_cuDoubleComplex(1.0,0);
		                tpsi1=cuCsub(tpsi2,tpsi1);
						tpsi1=cuCmul(tpsi1,d_Psi[col][row]);
		cuDoubleComplex tpsix3=cuCmul(d_Ux[col][row],d_Psi[col+1][row]);
		cuDoubleComplex tpsix4=cuCmul(cuConj(d_Ux[col-1][row]),d_Psi[col-1][row]);
						tpsix3=cuCadd(tpsix3,tpsix4);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
        cuDoubleComplex tpsiy3=cuCmul(d_Uy[col][row],d_Psi[col][row+1]);
		cuDoubleComplex tpsiy4=cuCmul(cuConj(d_Uy[col][row-1]),d_Psi[col][row-1]);
						tpsiy3=cuCadd(tpsiy3,tpsiy4);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
        cuDoubleComplex tpsix5=make_cuDoubleComplex(d_dx*d_dx*kappa*kappa,0.0);
		cuDoubleComplex tpsiy5=make_cuDoubleComplex(d_dy*d_dy*kappa*kappa,0.0);
		cuDoubleComplex tpsix=cuCdiv(tpsix3,tpsix5);
		cuDoubleComplex tpsiy=cuCdiv(tpsiy3,tpsiy5);
						d_dPsidt[col][row]=cuCadd(tpsi1,tpsix);
						d_dPsidt[col][row]=cuCadd(d_dPsidt[col][row],tpsiy);
	}
	if(row==UpperBoun1+1&&((col>=LeftBoun1&&col<=RightBoun1)||(col>=LeftBoun2&&col<=RightBoun2))){//calculate the dpsi/dt of the upper side of the hole boundary//put 0.9 to push left boundary one mesh forward 
	    tempPsi1[col][row-1]=cuCmul(d_Uy[col][row-1],d_Psi[col][row]);
//	    cuDoubleComplex tjx1=cuCmul(d_Ux[col][row-1],cuConj(d_Psi[col][row-1]));
//		                tjx1=cuCmul(tjx1,d_Psi[col+1][row-1]);
//		cuDoubleComplex tjx2=make_cuDoubleComplex(d_dx*kappa,0.0);
//      d_Jsx[col][row-1]=cuCimag(cuCdiv(tjx1,tjx2));//calculate the Jsx
	    cuDoubleComplex tpsi1=cuCmul(d_Psi[col][row],cuConj(d_Psi[col][row]));
		cuDoubleComplex tpsi2=make_cuDoubleComplex(1.0,0);
		                tpsi1=cuCsub(tpsi2,tpsi1);
						tpsi1=cuCmul(tpsi1,d_Psi[col][row]);
		cuDoubleComplex tpsix3=cuCmul(d_Ux[col][row],d_Psi[col+1][row]);
		cuDoubleComplex tpsix4=cuCmul(cuConj(d_Ux[col-1][row]),d_Psi[col-1][row]);
						tpsix3=cuCadd(tpsix3,tpsix4);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
        cuDoubleComplex tpsiy3=cuCmul(d_Uy[col][row],d_Psi[col][row+1]);
		cuDoubleComplex tpsiy4=cuCmul(cuConj(d_Uy[col][row-1]),tempPsi1[col][row-1]);
						tpsiy3=cuCadd(tpsiy3,tpsiy4);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
        cuDoubleComplex tpsix5=make_cuDoubleComplex(d_dx*d_dx*kappa*kappa,0.0);
		cuDoubleComplex tpsiy5=make_cuDoubleComplex(d_dy*d_dy*kappa*kappa,0.0);
		cuDoubleComplex tpsix=cuCdiv(tpsix3,tpsix5);
		cuDoubleComplex tpsiy=cuCdiv(tpsiy3,tpsiy5);
						d_dPsidt[col][row]=cuCadd(tpsi1,tpsix);
						d_dPsidt[col][row]=cuCadd(d_dPsidt[col][row],tpsiy);
	}
	if(row==LowerBoun1-1&&((col>=LeftBoun1&&col<=RightBoun1)||(col>=LeftBoun2&&col<=RightBoun2))){//calculate the dpsi/dt of the lower side of the hole boundary
		tempPsi1[col][row+1]=cuCmul(cuConj(d_Uy[col][row]),d_Psi[col][row]);
		cuDoubleComplex tjy1=cuCmul(d_Uy[col][row],cuConj(d_Psi[col][row]));
		                tjy1=cuCmul(tjy1,tempPsi1[col][row+1]);
		cuDoubleComplex tjy2=make_cuDoubleComplex(d_dy*kappa,0.0);
		d_Jsy[col][row]=cuCimag(cuCdiv(tjy1,tjy2));
//	    cuDoubleComplex tjx1=cuCmul(d_Ux[col][row+1],cuConj(d_Psi[col][row+1]));
//		                tjx1=cuCmul(tjx1,d_Psi[col+1][row+1]);
//		cuDoubleComplex tjx2=make_cuDoubleComplex(d_dx*kappa,0.0);
//        d_Jsx[col][row+1]=cuCimag(cuCdiv(tjx1,tjx2));//calculate the Jsx at boundary(actually for the convennience of corner point)
	    cuDoubleComplex tpsi1=cuCmul(d_Psi[col][row],cuConj(d_Psi[col][row]));
		cuDoubleComplex tpsi2=make_cuDoubleComplex(1.0,0);
		                tpsi1=cuCsub(tpsi2,tpsi1);
						tpsi1=cuCmul(tpsi1,d_Psi[col][row]);
		cuDoubleComplex tpsix3=cuCmul(d_Ux[col][row],d_Psi[col+1][row]);
		cuDoubleComplex tpsix4=cuCmul(cuConj(d_Ux[col-1][row]),d_Psi[col-1][row]);
						tpsix3=cuCadd(tpsix3,tpsix4);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
        cuDoubleComplex tpsiy3=cuCmul(d_Uy[col][row],tempPsi1[col][row+1]);
		cuDoubleComplex tpsiy4=cuCmul(cuConj(d_Uy[col][row-1]),d_Psi[col][row-1]);
						tpsiy3=cuCadd(tpsiy3,tpsiy4);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
        cuDoubleComplex tpsix5=make_cuDoubleComplex(d_dx*d_dx*kappa*kappa,0.0);
		cuDoubleComplex tpsiy5=make_cuDoubleComplex(d_dy*d_dy*kappa*kappa,0.0);
		cuDoubleComplex tpsix=cuCdiv(tpsix3,tpsix5);
		cuDoubleComplex tpsiy=cuCdiv(tpsiy3,tpsiy5);
						d_dPsidt[col][row]=cuCadd(tpsi1,tpsix);
						d_dPsidt[col][row]=cuCadd(d_dPsidt[col][row],tpsiy);
	}
    if((col==RightBoun1+1||col==RightBoun2+1)&&row>=LowerBoun1&&row<=UpperBoun1){//calculate the dpsi/dt of the right side of the hole boundary
		d_Psi[col-1][row]=cuCmul(d_Ux[col-1][row],d_Psi[col][row]);
		cuDoubleComplex tjy1=cuCmul(d_Uy[col-1][row],cuConj(d_Psi[col-1][row]));
		                tjy1=cuCmul(tjy1,d_Psi[col-1][row+1]);
		cuDoubleComplex tjy2=make_cuDoubleComplex(d_dy*kappa,0.0);
		d_Jsy[col-1][row]=cuCimag(cuCdiv(tjy1,tjy2));
	    cuDoubleComplex tpsi1=cuCmul(d_Psi[col][row],cuConj(d_Psi[col][row]));
		cuDoubleComplex tpsi2=make_cuDoubleComplex(1.0,0);
		                tpsi1=cuCsub(tpsi2,tpsi1);
						tpsi1=cuCmul(tpsi1,d_Psi[col][row]);
		cuDoubleComplex tpsix3=cuCmul(d_Ux[col][row],d_Psi[col+1][row]);
		cuDoubleComplex tpsix4=cuCmul(cuConj(d_Ux[col-1][row]),d_Psi[col-1][row]);
						tpsix3=cuCadd(tpsix3,tpsix4);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
        cuDoubleComplex tpsiy3=cuCmul(d_Uy[col][row],d_Psi[col][row+1]);
		cuDoubleComplex tpsiy4=cuCmul(cuConj(d_Uy[col][row-1]),d_Psi[col][row-1]);
						tpsiy3=cuCadd(tpsiy3,tpsiy4);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
        cuDoubleComplex tpsix5=make_cuDoubleComplex(d_dx*d_dx*kappa*kappa,0.0);
		cuDoubleComplex tpsiy5=make_cuDoubleComplex(d_dy*d_dy*kappa*kappa,0.0);
		cuDoubleComplex tpsix=cuCdiv(tpsix3,tpsix5);
		cuDoubleComplex tpsiy=cuCdiv(tpsiy3,tpsiy5);
						d_dPsidt[col][row]=cuCadd(tpsi1,tpsix);
						d_dPsidt[col][row]=cuCadd(d_dPsidt[col][row],tpsiy);
	}
    if((col==LeftBoun1-1||col==LeftBoun2-1)&&row>=LowerBoun1&&row<=UpperBoun1){//calculate the dpsi/dt of the left side of the hole boundary
		d_Psi[col+1][row]=cuCmul(cuConj(d_Ux[col][row]),d_Psi[col][row]);
	    cuDoubleComplex tjx1=cuCmul(d_Ux[col][row],cuConj(d_Psi[col][row]));
		                tjx1=cuCmul(tjx1,d_Psi[col+1][row]);
		cuDoubleComplex tjx2=make_cuDoubleComplex(d_dx*kappa,0.0);
        d_Jsx[col][row]=cuCimag(cuCdiv(tjx1,tjx2));
		cuDoubleComplex tjy1=cuCmul(d_Uy[col+1][row],cuConj(d_Psi[col+1][row]));
		                tjy1=cuCmul(tjy1,d_Psi[col+1][row+1]);
		cuDoubleComplex tjy2=make_cuDoubleComplex(d_dy*kappa,0.0);
		d_Jsy[col+1][row]=cuCimag(cuCdiv(tjy1,tjy2));
	    cuDoubleComplex tpsi1=cuCmul(d_Psi[col][row],cuConj(d_Psi[col][row]));
		cuDoubleComplex tpsi2=make_cuDoubleComplex(1.0,0);
		                tpsi1=cuCsub(tpsi2,tpsi1);
						tpsi1=cuCmul(tpsi1,d_Psi[col][row]);
		cuDoubleComplex tpsix3=cuCmul(d_Ux[col][row],d_Psi[col+1][row]);
		cuDoubleComplex tpsix4=cuCmul(cuConj(d_Ux[col-1][row]),d_Psi[col-1][row]);
						tpsix3=cuCadd(tpsix3,tpsix4);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
						tpsix3=cuCsub(tpsix3,d_Psi[col][row]);
        cuDoubleComplex tpsiy3=cuCmul(d_Uy[col][row],d_Psi[col][row+1]);
		cuDoubleComplex tpsiy4=cuCmul(cuConj(d_Uy[col][row-1]),d_Psi[col][row-1]);
						tpsiy3=cuCadd(tpsiy3,tpsiy4);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
						tpsiy3=cuCsub(tpsiy3,d_Psi[col][row]);
        cuDoubleComplex tpsix5=make_cuDoubleComplex(d_dx*d_dx*kappa*kappa,0.0);
		cuDoubleComplex tpsiy5=make_cuDoubleComplex(d_dy*d_dy*kappa*kappa,0.0);
		cuDoubleComplex tpsix=cuCdiv(tpsix3,tpsix5);
		cuDoubleComplex tpsiy=cuCdiv(tpsiy3,tpsiy5);
						d_dPsidt[col][row]=cuCadd(tpsi1,tpsix);
						d_dPsidt[col][row]=cuCadd(d_dPsidt[col][row],tpsiy);
	}
//	tempPsi1[col][row]=d_Psi[col][row];
//	tempPsi2[col][row]=d_Psi[col][row];
}

__device__ void CUDATDGL::CUcaldUdt(int row, int col){
	if(col<Nx&&row>0&&row<Ny){//calculate dUx/dt and the value of hole-corner points are decided by left and right boundary condition 
        cuDoubleComplex tx1=make_cuDoubleComplex(d_dy*d_dy,0);
	    cuDoubleComplex tx2=cuCsub(d_W[col][row],d_W[col][row-1]);
		                tx2=cuCdiv(tx2,tx1);
		cuDoubleComplex tx3=cuCmul(d_Ux[col][row],cuConj(d_Psi[col][row]));
		                tx3=cuCmul(tx3,d_Psi[col+1][row]);
	    d_ImFux[col][row]=cuCimag(cuCadd(tx2,tx3));//cuCimag(cuCadd(tx2,tx3));
		if((row==LowerBoun1||row==UpperBoun1)&&col>=LeftBoun1&&col<=RightBoun1-1){
		cuDoubleComplex tx3=cuCmul(d_Ux[col][row],cuConj(tempPsi1[col][row]));
		                tx3=cuCmul(tx3,tempPsi1[col+1][row]);
	    d_ImFux[col][row]=cuCimag(cuCadd(tx2,tx3));//cuCimag(cuCadd(tx2,tx3));
		}
		if((row==LowerBoun1||row==UpperBoun1)&&col>=LeftBoun2&&col<=RightBoun2-1){
		cuDoubleComplex tx3=cuCmul(d_Ux[col][row],cuConj(tempPsi1[col][row]));
		                tx3=cuCmul(tx3,tempPsi1[col+1][row]);
	    d_ImFux[col][row]=cuCimag(cuCadd(tx2,tx3));//cuCimag(cuCadd(tx2,tx3));
		}
	}
	if(col>0&&row<Ny&&col<Nx){//calculate dUy/dt
	    cuDoubleComplex ty1=make_cuDoubleComplex(d_dx*d_dx,0);
		cuDoubleComplex ty2=cuCsub(d_W[col-1][row],d_W[col][row]);
		                ty2=cuCdiv(ty2,ty1);
		cuDoubleComplex ty3=cuCmul(d_Uy[col][row],cuConj(tempPsi1[col][row]));
			            ty3=cuCmul(ty3,tempPsi1[col][row+1]);
	    d_ImFuy[col][row]=cuCimag(cuCadd(ty2,ty3));
		if((col==LeftBoun1||col==RightBoun1||col==LeftBoun2||col==RightBoun2)&&row<=UpperBoun1-1&&row>=LowerBoun1){
		cuDoubleComplex ty3=cuCmul(d_Uy[col][row],cuConj(d_Psi[col][row]));
			            ty3=cuCmul(ty3,d_Psi[col][row+1]);
	    d_ImFuy[col][row]=cuCimag(cuCadd(ty2,ty3));		
		}
	}
	if(col==Nx&&row==Ny)
		Totaltime++;
}


//calculate the boundary current with Js*n=0, square hole length (Ny+1)/2

__device__ void CUDATDGL::CUcalBC(int row,int col){
	cuDoubleComplex tc1=make_cuDoubleComplex(0.0, 0.0);
	tempPsi1[col][row]=tc1;
	if(row>=LowerBoun1&&row<=UpperBoun1&&col>=LeftBoun1&&col<=RightBoun1){//build the two holes
	    d_Psi[col][row]=tc1;
	}
	if(row>=LowerBoun2&&row<=UpperBoun2&&col>=LeftBoun2&&col<=RightBoun2){
	    d_Psi[col][row]=tc1;
	}
	if(row==0&&col>0&&col<Nx){
		d_Psi[col][row]=cuCmul(d_Uy[col][row],d_Psi[col][row+1]);
	}
	if(row==Ny&&col>0&&col<Nx){
		d_Psi[col][row]=cuCmul(cuConj(d_Uy[col][row-1]),d_Psi[col][row-1]);
	}
	if(col==0&&row>0&&row<Ny){
		d_Psi[col][row]=cuCmul(d_Ux[col][row],d_Psi[col+1][row]);
	}
	if(col==Nx&&row>0&&row<Ny){
		d_Psi[col][row]=cuCmul(cuConj(d_Ux[col-1][row]),d_Psi[col-1][row]);
	}
//	if(col==Nx&&row==Ny&&Totaltime==12000*100){
//	    LeftBoun1++;
//		RightBoun1++;
//		LeftBoun2--;
//		RightBoun2--;
//	}
//	if(row==int(3*Ny/4+0.01)&&col>Nx/2-Ny/4&&col<Nx/2+Ny/4-0.99){//calculate the boundary condition of the hole but ignore the corner point
//		d_Psi[col][row]=cuCmul(d_Uy[col][row],d_Psi[col][row+1]);
//	}
//	if(row==int(Ny/4+0.01)&&col>Nx/2-Ny/4&&col<Nx/2+Ny/4-0.99){
//		d_Psi[col][row]=cuCmul(cuConj(d_Uy[col][row-1]),d_Psi[col][row-1]);
//	}
//	if(col==int(Nx/2+Ny/4+0.01)&&row>Ny/4&&row<3*Ny/4-0.99){
//		d_Psi[col][row]=cuCmul(d_Ux[col][row],d_Psi[col+1][row]);
//	}
//	if(col==int(Nx/2-Ny/4+0.01)&&row>Ny/4&&row<3*Ny/4-0.99){
//		d_Psi[col][row]=cuCmul(cuConj(d_Ux[col-1][row]),d_Psi[col-1][row]);
//	}
}


__device__ void CUDATDGL::CUonestep(int row,int col){
    double faix=-d_dt*d_ImFux[col][row]/sigma;
	cuDoubleComplex tx1=make_cuDoubleComplex(cos(faix),sin(faix));
	d_Ux[col][row]=cuCmul(d_Ux[col][row],tx1);
	cuDoubleComplex tx2=make_cuDoubleComplex(cuCabs(d_Ux[col][row]),0.0);
	d_Ux[col][row]=cuCdiv(d_Ux[col][row],tx2);
	double faiy=-d_dt*d_ImFuy[col][row]/sigma;
	cuDoubleComplex ty1=make_cuDoubleComplex(cos(faiy),sin(faiy));
    d_Uy[col][row]=cuCmul(d_Uy[col][row],ty1);
	cuDoubleComplex ty2=make_cuDoubleComplex(cuCabs(d_Uy[col][row]),0.0);
	d_Uy[col][row]=cuCdiv(d_Uy[col][row],ty2);
	if(!((row>=LowerBoun1&&row<=UpperBoun1&&col>=LeftBoun1&&col<=RightBoun1)||\
		(row>=LowerBoun2&&row<=UpperBoun2&&col>=LeftBoun2&&col<=RightBoun2))){
	    cuDoubleComplex ct1=make_cuDoubleComplex(d_dt,0);
                        ct1=cuCmul(d_dPsidt[col][row],ct1);
	    d_Psi[col][row]=cuCadd(d_Psi[col][row],ct1);
	}
}

__device__ void CUDATDGL::CUcalJs(int row,int col){
	if(col<Nx&&row<Ny){
	    cuDoubleComplex tjx1=cuCmul(d_Ux[col][row],cuConj(d_Psi[col][row]));
		                tjx1=cuCmul(tjx1,d_Psi[col+1][row]);
		cuDoubleComplex tjx2=make_cuDoubleComplex(d_dx*kappa,0.0);
        d_Jsx[col][row]=cuCimag(cuCdiv(tjx1,tjx2));
		cuDoubleComplex tjy1=cuCmul(d_Uy[col][row],cuConj(d_Psi[col][row]));
		                tjy1=cuCmul(tjy1,d_Psi[col][row+1]);
		cuDoubleComplex tjy2=make_cuDoubleComplex(d_dy*kappa,0.0);
		d_Jsy[col][row]=cuCimag(cuCdiv(tjy1,tjy2));
	}
}

__device__ void CUDATDGL::CUcalNumOfVortices(int row,int col){
	d_VOR[col][row]=0.0;
	d_Ax[col][row]=0.0;
	d_Ay[col][row]=0.0;
	int d_width=Nx+1,d_height=Ny+1;
    double d_sum=0.0,temp;
	if((row==1||row==Ny-1)&&col<Nx&&col>0){
        cuDoubleComplex tjx1=cuCmul(d_Ux[col][row],cuConj(d_Psi[col][row]));
		                tjx1=cuCmul(tjx1,d_Psi[col+1][row]);
		cuDoubleComplex tjx2=make_cuDoubleComplex(d_dx*kappa,0.0);
        d_Jsx[col][row]=cuCimag(cuCdiv(tjx1,tjx2));
		double ta1=-cuCimag(d_Ux[col][row])/cuCabs(d_Ux[col][row]);
		d_Ax[col][row]=asin(ta1)/kappa/d_dx;
		cuDoubleComplex tax=cuCmul(d_Psi[col][row],cuConj(d_Psi[col][row]));
		d_VOR[col][row]=(d_Ax[col][row])*d_dx;
		d_VOR[col][row]=(d_Jsx[col][row]/cuCreal(tax)+d_Ax[col][row])*d_dx;//preparation for CPU to calculate the sum.
	}
	if((col==1||col==Nx-1)&&row<Ny&&row>0){
	    cuDoubleComplex tjy1=cuCmul(d_Uy[col][row],cuConj(d_Psi[col][row]));
		                tjy1=cuCmul(tjy1,d_Psi[col][row+1]);
		cuDoubleComplex tjy2=make_cuDoubleComplex(d_dy*kappa,0.0);
		d_Jsy[col][row]=cuCimag(cuCdiv(tjy1,tjy2));
		double ta1=-cuCimag(d_Uy[col][row])/cuCabs(d_Uy[col][row]);
		d_Ay[col][row]=asin(ta1)/kappa/d_dy;
		cuDoubleComplex tay=cuCmul(d_Psi[col][row],cuConj(d_Psi[col][row]));
		d_VOR[col][row]=(d_Ay[col][row])*d_dy;
		d_VOR[col][row]=(d_Jsy[col][row]/cuCreal(tay)+d_Ay[col][row])*d_dy;//preparation for CPU to calculate the sum.
	}
}


__device__ void CUDATDGL::CUcalSysEng(int row,int col,double *d_Ba){
    double CondEng=0.0,KEng=0.0,MagEng=0.0;
	double temp;
	if(row>0&&row<Ny&&col>0&&col<Nx){
		if((!(row>=LowerBoun1&&row<=UpperBoun1&&col>=LeftBoun1&&col<=RightBoun1))\
			&&(!(row>=LowerBoun2&&row<=UpperBoun2&&col>=LeftBoun2&&col<=RightBoun2))){
            temp=cuCreal(cuCmul(d_Psi[col][row],cuConj(d_Psi[col][row])));
	        CondEng=temp*temp/2-temp;
		}
		if((!(row>=LowerBoun1&&row<=UpperBoun1&&col>=LeftBoun1-1&&col<=RightBoun1))\
			&&(!(row>=LowerBoun2&&row<=UpperBoun2&&col>=LeftBoun2-1&&col<=RightBoun2))){
	        cuDoubleComplex tc1=cuCsub(cuCmul(d_Ux[col][row],d_Psi[col+1][row]),d_Psi[col][row]);
		    KEng=KEng+cuCreal(cuCmul(tc1,cuConj(tc1)))/kappa/kappa/d_dx/d_dx;
		}
		if((!(row>=LowerBoun1-1&&row<=UpperBoun1&&col>=LeftBoun1&&col<=RightBoun1))\
			&&(!(row>=LowerBoun2-1&&row<=UpperBoun2&&col>=LeftBoun2&&col<=RightBoun2))){
            cuDoubleComplex tc2=cuCsub(cuCmul(d_Uy[col][row],tempPsi1[col][row+1]),tempPsi1[col][row]);
		    KEng=KEng+cuCreal(cuCmul(tc2,cuConj(tc2)))/kappa/kappa/d_dy/d_dy;
		}
	}
		cuDoubleComplex tbz1=make_cuDoubleComplex(1.0,0.0);
		cuDoubleComplex tbz2=make_cuDoubleComplex(kappa*d_dx*d_dy,0.0);
                        tbz1=cuCsub(tbz1,d_W[col][row]);
                        tbz1=cuCdiv(tbz1,tbz2);
		d_Bz[col][row]=cuCimag(tbz1);
		MagEng=MagEng+d_Bz[col][row]*d_Bz[col][row]-2*d_Bz[col][row]*(*d_Ba);//+2*d_Bz[col][row]*2.6;
		d_ENG[col][row]=(MagEng+KEng+CondEng)*d_dx*d_dy;
}


void HOSTTDGL::OutputPsi(int fileid){
	int BA;
	BA=int(nameBa*100+0.1);
	ini_par();
	double temp;
	string fn("Psi2");
	char c[5];
	sprintf(c,"%d",BA);
	fn.append(c);
	fn.append("_");
	char d[5];
	sprintf(d,"%d",fileid);
	fn.append(d);
	fn.append(".dat");
    FILE *fp=fopen(fn.c_str(),"w");
	fprintf(fp, "%g", 0.0);
	for(int i=0; i<Nx+1; i++){
		fprintf(fp, "\t%g", i*dx);
	}
	fprintf(fp,"\n");
	for(int j=0;j<Ny+1;j++){
		fprintf(fp, "%g", j*dy);
		for(int i=0;i<Nx+1;i++){
		    temp=cuCabs(Psi[i][j]);
			fprintf(fp,"\t%8.8lf",temp);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void HOSTTDGL::OutputBz(int fileid){
	int BA;
	BA=int(nameBa*100+0.1);
	ini_par();
	double temp;
	string fn("Bz2");
	char c[5];
	sprintf(c,"%d",BA);
	fn.append(c);
	fn.append("_");
	char d[5];
	sprintf(d,"%d",fileid);
	fn.append(d);
	fn.append(".dat");
	FILE *fp=fopen(fn.c_str(),"w");
	fprintf(fp, "%g", 0.0);
	for(int i=0; i<Nx+1; i++){
		fprintf(fp, "\t%g", i*dx);
	}
	fprintf(fp,"\n");
	for(int j=0;j<Ny+1;j++){
		fprintf(fp, "%g", j*dy);
		for(int i=0;i<Nx+1;i++){
		    temp=Bz[i][j];
			fprintf(fp,"\t%8.8lf",temp);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void HOSTTDGL::OutputPsiPhase(int fileid){
	int BA;
	BA=int(nameBa*100+0.1);
	ini_par();
	double temp;
	string fn("PsiPhase2");
	char c[5];
	sprintf(c,"%d",BA);
	fn.append(c);
	fn.append("_");
	char d[5];
	sprintf(d,"%d",fileid);
	fn.append(d);
	fn.append(".dat");
	FILE *fp=fopen(fn.c_str(),"w");
	fprintf(fp, "%g", 0.0);
	for(int i=0; i<Nx+1; i++){
		fprintf(fp, "\t%g", i*dx);
	}
	fprintf(fp,"\n");
	for(int j=0;j<Ny+1;j++){
		fprintf(fp, "%g", j*dy);
		for(int i=0;i<Nx+1;i++){
		    temp=cuCreal(Psi[i][j])/cuCabs(Psi[i][j]);
			fprintf(fp,"\t%lf",temp);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void HOSTTDGL::OutputJsy(int fileid){
	int BA;
	BA=int(nameBa*100+0.1);
	ini_par();
	double temp;
	string fn("Jsy2");
	char c[5];
	sprintf(c,"%d",BA);
	fn.append(c);
	fn.append("_");
	char d[5];
	sprintf(d,"%d",fileid);
	fn.append(d);
	fn.append(".dat");
	FILE *fp=fopen(fn.c_str(),"w");
	fprintf(fp, "%g", 0.0);
	for(int i=0; i<Nx+1; i++){
		fprintf(fp, "\t%g", i*dx);
	}
	fprintf(fp,"\n");
	for(int j=0;j<Ny+1;j++){
		fprintf(fp, "%g", j*dy);
		for(int i=0;i<Nx+1;i++){
		    temp=Jsy[i][j];
			fprintf(fp,"\t%lf",temp);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void HOSTTDGL::OutputJsx(int fileid){
	int BA;
	BA=int(nameBa*100+0.1);
	ini_par();
	double temp;
	string fn("Jsx2");
	char c[5];
	sprintf(c,"%d",BA);
	fn.append(c);
	fn.append("_");
	char d[5];
	sprintf(d,"%d",fileid);
	fn.append(d);
	fn.append(".dat");
	FILE *fp=fopen(fn.c_str(),"w");
	fprintf(fp, "%g", 0.0);
	for(int i=0; i<Nx+1; i++){
		fprintf(fp, "\t%g", i*dx);
	}
	fprintf(fp,"\n");
	for(int j=0;j<Ny+1;j++){
		fprintf(fp, "%g", j*dy);
		for(int i=0;i<Nx+1;i++){
		    temp=Jsx[i][j];
			fprintf(fp,"\t%8.8lf",temp);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void HOSTTDGL::OutputAx(int fileid){
	int BA;
	BA=int(nameBa*100+0.1);
	ini_par();
	double temp;
	string fn("ENG2");
	char c[5];
	sprintf(c,"%d",BA);
	fn.append(c);
	fn.append("_");
	char d[5];
	sprintf(d,"%d",fileid);
	fn.append(d);
	fn.append(".dat");
	FILE *fp=fopen(fn.c_str(),"w");
	fprintf(fp, "%g", 0.0);
	for(int i=0; i<Nx+1; i++){
		fprintf(fp, "\t%g", i*dx);
	}
	fprintf(fp,"\n");
	for(int j=0;j<Ny+1;j++){
		fprintf(fp, "%g", j*dy);
		for(int i=0;i<Nx+1;i++){
		    temp=ENG[i][j];
			fprintf(fp,"\t%lf",temp);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void HOSTTDGL::OutputUx(int fileid){
	int BA;
	BA=int(nameBa*100+0.1);
	ini_par();
	double temp;
	string fn("Ux2");
	char c[5];
	sprintf(c,"%d",BA);
	fn.append(c);
	fn.append("_");
	char d[5];
	sprintf(d,"%d",fileid);
	fn.append(d);
	fn.append(".dat");
	FILE *fp=fopen(fn.c_str(),"w");
	fprintf(fp, "%g", 0.0);
	for(int i=0; i<Nx+1; i++){
		fprintf(fp, "\t%g", i*dx);
	}
	fprintf(fp,"\n");
	for(int j=0;j<Ny+1;j++){
		fprintf(fp, "%g", j*dy);
		for(int i=0;i<Nx+1;i++){
		    temp=cuCimag(Psi[i][j])/cuCabs(Psi[i][j]);
			fprintf(fp,"\t%8.8lf",temp);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}


void HOSTTDGL::CalTotal(double Ba){
	count=0;
	NumOfVortices=0.0;
	SysEng=0.0;
	magnetization=0.0;
	for(int i=1;i<Nx-1;i++){
	    int j=1;
		NumOfVortices+=VOR[i][j];
		j=Ny-1;
		NumOfVortices-=VOR[i][j];
	}
	for(int j=1;j<Ny-1;j++){
	    int i=1;
		NumOfVortices-=VOR[i][j];
		i=Nx-1;
		NumOfVortices+=VOR[i][j];
	}
	NumOfVortices=NumOfVortices*kappa/6.2831852;
	for(int i=0;i<Nx;i++){
		for(int j=0;j<Ny;j++){
		        SysEng+=ENG[i][j];
			if(!((j>=int(Ny/4.0+0.9-ExpandSteps)&&j<=int(3.0*Ny/4.0+0.01+ExpandSteps)&&i>=int(Ny/4.0+ApproachingSteps+0.9-ExpandSteps)&&i<=int(3.0*Ny/4.0+ApproachingSteps+0.01+ExpandSteps))||\
				(j>=int(Ny/4.0+0.9-ExpandSteps)&&j<=int(3.0*Ny/4.0+0.01+ExpandSteps)&&i>=int(Nx/2+Ny/4.0-ApproachingSteps+0.9-ExpandSteps)&&i<=int(Nx/2+3.0*Ny/4.0-ApproachingSteps+0.01+ExpandSteps)))){
				count++;
			    magnetization+=Bz[i][j];
			}
		}
	}
	magnetization=magnetization/count;
	magnetization-=Ba;
}
