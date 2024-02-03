#include "cuTDGL.h"
#include "cuComplex.h"
#include <iostream>
#include "math.h"

__global__ void CUDAinitMesh();
__global__ void CUDAcalW(double *d_Ba);
__global__ void CUDAcalBC();
__global__ void CUDAcaldPsidt();
__global__ void CUDAcaldUdt();
__global__ void CUDAonestep();
__global__ void CUDACalTotal(double *d_Ba);
__global__ void TDGLoutput(cuDoubleComplex *dev_psi,double *dev_Jsx,double *dev_Jsy,double *dev_Ax,\
						   cuDoubleComplex *dev_Ux,double *dev_VOR,double *dev_ENG,double *dev_Bz);

int main(){
	cuDoubleComplex *dev_Psi;
	double *dev_Jsx;
	double *dev_Jsy;
	double *dev_Ax;
	cuDoubleComplex *dev_Ux;
	double *dev_VOR;
	double *dev_ENG;
	double *d_Ba;
	double *dev_Bz;
	width=Nx+1;
	height=Ny+1;
	sizeComplex=width*height*sizeof(cuDoubleComplex);
	sizeDouble=width*height*sizeof(double);
	cudaMalloc(&dev_Psi,sizeComplex);
	cudaMalloc(&dev_Jsx,sizeDouble);
	cudaMalloc(&dev_Jsy,sizeDouble);
	cudaMalloc(&dev_Ux,sizeComplex);
	cudaMalloc(&d_Ba,sizeof(double));
	cudaMalloc(&dev_VOR,sizeDouble);
	cudaMalloc(&dev_ENG,sizeDouble);
	cudaMalloc(&dev_Bz,sizeDouble);
	cudaMalloc(&dev_Ax,sizeDouble);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x,(height+dimBlock.y-1)/dimBlock.y);
	CUDAinitMesh<<<dimGrid,dimBlock>>>();//initialize the mesh
	cudaThreadSynchronize();
    
	int sample=100;
	int i,prepare=96700,fileid=0,total=50,n;
	prepare*=sample;
	total*=sample;
	double sBa=10.5,eBa=20.3,stepBa1=0.01,stepBa2=-0.5,nowBa=4.75,interBa=100;
	nowBa=sBa;
	double sumMag=0.0,sumNumVor=0.0,sumSysEng=0.0;
	FILE *fpMag=fopen("Mag2.dat","w");
	fprintf(fpMag,"Ba\tMag\tNumVor\tSysEng\n");
	fclose(fpMag);
	while(nowBa<eBa&&nowBa>=0.0){
		n=0;
        cudaMemcpy(d_Ba,&nowBa,sizeof(double),cudaMemcpyHostToDevice);
		double magtest[300];
	    for(i=0;i<prepare;i++){
	        CUDAcalBC<<<dimGrid,dimBlock>>>();
//	        cudaThreadSynchronize();
	        CUDAcalW<<<dimGrid,dimBlock>>>(d_Ba);
//	        cudaThreadSynchronize();
	        CUDAcaldPsidt<<<dimGrid,dimBlock>>>();
//	        cudaThreadSynchronize();
			CUDAcaldUdt<<<dimGrid,dimBlock>>>();
//			cudaThreadSynchronize();
	        CUDAonestep<<<dimGrid,dimBlock>>>();
//	        cudaThreadSynchronize();
//		}
		if(i%(300*sample)==0){
                CUDACalTotal<<<dimGrid,dimBlock>>>(d_Ba);
	            cudaThreadSynchronize();
			    TDGLoutput<<<dimGrid,dimBlock>>>(dev_Psi,dev_Jsx,dev_Jsy,dev_Ax,dev_Ux,dev_VOR,dev_ENG,dev_Bz);
				cudaThreadSynchronize();
	            cudaMemcpy(HGL.Psi,dev_Psi,sizeComplex,cudaMemcpyDeviceToHost);
	            cudaMemcpy(HGL.Jsx,dev_Jsx,sizeDouble,cudaMemcpyDeviceToHost);
	            cudaMemcpy(HGL.Jsy,dev_Jsy,sizeDouble,cudaMemcpyDeviceToHost);
	            cudaMemcpy(HGL.Ux,dev_Ux,sizeComplex,cudaMemcpyDeviceToHost);
				cudaMemcpy(HGL.VOR,dev_VOR,sizeDouble,cudaMemcpyDeviceToHost);
				cudaMemcpy(HGL.ENG,dev_ENG,sizeDouble,cudaMemcpyDeviceToHost);
				cudaMemcpy(HGL.Bz,dev_Bz,sizeDouble,cudaMemcpyDeviceToHost);
				cudaMemcpy(HGL.Ax,dev_Ax,sizeDouble,cudaMemcpyDeviceToHost);

	            fileid++; 

				HGL.CalTotal(nowBa);
	            printf("%lf\t%lf\t%8.8lf\t%lf\n", HGL.magnetization,HGL.NumOfVortices,HGL.SysEng,nowBa);
				magtest[n]=HGL.magnetization;
				HGL.nameBa=nowBa;//add Ba to the title of file's name to find the corresponding file more conveniently
	            HGL.OutputPsi(fileid);
	            HGL.OutputBz(fileid);
	            HGL.OutputJsy(fileid);
	            HGL.OutputJsx(fileid);
				HGL.OutputPsiPhase(fileid);
				HGL.OutputAx(fileid);
				HGL.OutputUx(fileid);
				int N=300;
				if(nowBa<12.15)
					N=100;
				if(n>N&&abs(magtest[n]-magtest[n-1])<0.000001&&abs(magtest[n]-magtest[n-20])<0.000001)//get out of the loop when equilibrium arrives
					break;
				n++;
			}
		}
		for(i=0;i<total;i++){
	        CUDAcalBC<<<dimGrid,dimBlock>>>();
	        //cudaThreadSynchronize();
	        CUDAcalW<<<dimGrid,dimBlock>>>(d_Ba);
	        //cudaThreadSynchronize();
	        CUDAcaldPsidt<<<dimGrid,dimBlock>>>();
	        //cudaThreadSynchronize();
			CUDAcaldUdt<<<dimGrid,dimBlock>>>();
			//cudaThreadSynchronize();
	        CUDAonestep<<<dimGrid,dimBlock>>>();
	        //cudaThreadSynchronize();
            if(i%sample==0){
                CUDACalTotal<<<dimGrid,dimBlock>>>(d_Ba);
	            cudaThreadSynchronize();
			    TDGLoutput<<<dimGrid,dimBlock>>>(dev_Psi,dev_Jsx,dev_Jsy,dev_Ax,dev_Ux,dev_VOR,dev_ENG,dev_Bz);
				cudaThreadSynchronize();
	            cudaMemcpy(HGL.Psi,dev_Psi,sizeComplex,cudaMemcpyDeviceToHost);
//	            cudaMemcpy(HGL.Jsx,dev_Jsx,sizeComplex,cudaMemcpyDeviceToHost);
//	            cudaMemcpy(HGL.Jsy,dev_Jsy,sizeComplex,cudaMemcpyDeviceToHost);
//	            cudaMemcpy(HGL.W,dev_W,sizeComplex,cudaMemcpyDeviceToHost);
				cudaMemcpy(HGL.VOR,dev_VOR,sizeDouble,cudaMemcpyDeviceToHost);
				cudaMemcpy(HGL.ENG,dev_ENG,sizeDouble,cudaMemcpyDeviceToHost);
				cudaMemcpy(HGL.Bz,dev_Bz,sizeDouble,cudaMemcpyDeviceToHost);
				HGL.CalTotal(nowBa);
				sumMag+=HGL.magnetization;
				sumNumVor+=HGL.NumOfVortices;
				sumSysEng+=HGL.SysEng;
			}
		}
		sumSysEng=sumSysEng/(total/sample);
		sumMag=sumMag/(total/sample);
		sumNumVor=sumNumVor/(total/sample);
//		if(abs(sumMag)<0.0007)
//			interBa=nowBa-0.05;
		fpMag=fopen("Mag2.dat","a");
		fprintf(fpMag,"%lf\t%lf\t%lf\t%lf\n",nowBa,sumMag,sumNumVor,sumSysEng);
		fclose(fpMag);
		if(nowBa<interBa)
		    nowBa+=stepBa2;
		else
			nowBa+=stepBa1;
		sumSysEng=0.0;
        sumMag=0.0;
		sumNumVor=0.0;//change sumNumVor back to zero
	}
	cudaFree(dev_Psi);
	cudaFree(dev_Jsx);
	cudaFree(dev_Jsy);
	cudaFree(dev_Ux);
	cudaFree(dev_VOR);
	cudaFree(dev_ENG);
	cudaFree(dev_Bz);
	cudaFree(dev_Ax);
	return 1;
}


__global__ void CUDAinitMesh(){//initialize the parameters in mesh
    int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<Ny+1&&col<Nx+1)
        CGL.CUinitMesh(row,col);
}

__global__ void CUDAcalBC(){//calculate the boundary condition
    int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<Ny+1&&col<Nx+1)
	    CGL.CUcalBC(row,col);
}

__global__ void CUDAcalW(double *d_Ba){//calculate W which will be used to calculate dUi/dt
    int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<Ny+1&&col<Nx+1)
	    CGL.CUcalW(row,col,d_Ba);
}

__global__ void CUDAcaldPsidt(){//calculate dUi/dt and dPsi/dt
    int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<Ny+1&&col<Nx+1)
	    CGL.CUcaldPsidt(row,col);
}

__global__ void CUDAcaldUdt(){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<Ny+1&&col<Nx+1)
        CGL.CUcaldUdt(row,col);
}

__global__ void CUDAonestep(){//onestep forward
    int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<Ny+1&&col<Nx+1)
	    CGL.CUonestep(row,col);
}

__global__ void CUDACalTotal(double *d_Ba){//calculate the number of vortices and system energy per point
    int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<Ny+1&&col<Nx+1){
	    CGL.CUcalNumOfVortices(row,col);
	    CGL.CUcalSysEng(row,col,d_Ba);
//		CGL.CUcalBC(row,col);
		CGL.CUcalJs(row,col);
		CGL.CUcaldPsidt(row,col);//need to calculate the boundary current to cover the Js calculated just now
	}
}


__global__ void TDGLoutput(cuDoubleComplex *dev_Psi,double *dev_Jsx,double *dev_Jsy,double *dev_Ax,\
						   cuDoubleComplex *dev_Ux,double *dev_VOR,double *dev_ENG,double *dev_Bz){//output the datas you choose
    int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	int d_width=Ny+1;
	if(row<Ny+1&&col<Nx+1){
	    dev_Psi[col*d_width+row]=CGL.d_Psi[col][row];
	    dev_Jsx[col*d_width+row]=CGL.d_Jsx[col][row];
	    dev_Jsy[col*d_width+row]=CGL.d_Jsy[col][row];
	    dev_Ux[col*d_width+row]=CGL.d_Ux[col][row];
	    dev_VOR[col*d_width+row]=CGL.d_VOR[col][row];
	    dev_ENG[col*d_width+row]=CGL.d_ENG[col][row];
	    dev_Bz[col*d_width+row]=CGL.d_Bz[col][row];
		dev_Ax[col*d_width+row]=CGL.d_Ax[col][row];
	}
}
