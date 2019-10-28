#include "Header.h"


__global__ void moveKernel(Point* points, int N, double dT)
{
	const int i = blockIdx.x * MAX_NUM_OF_THREADS + threadIdx.x;
	if (i < N)
	{
		points[i].x += dT * points[i].Vx;
		points[i].y += dT * points[i].Vy;
		points[i].clusterID = 0;
	}
}

__global__ void groupKernel(Cluster* clusters, int K, Point* points, int N, char* flag)
{
	const int Pi = blockIdx.x * MAX_NUM_OF_THREADS + threadIdx.x;
	int Ci;
	double newDistance, oldDistance;
	//chande to redu[K][3][N] to overcome parallel ++ bug
	if(Pi  < N)
	{
		for (Ci = 0; Ci < K; Ci++)
		{
			
				oldDistance = sqrt(pow(points[Pi].x - clusters[points[Pi].clusterID].centerX, 2) + pow(points[Pi].y - clusters[points[Pi].clusterID].centerY, 2));
				newDistance = sqrt(pow(points[Pi].x - clusters[Ci].centerX, 2) + pow(points[Pi].y - clusters[Ci].centerY, 2));
				//calculate distances of points from cluster centers and make switch if necessery
				if (newDistance < oldDistance)
				{
					*flag = 1;
					points[Pi].clusterID = clusters[Ci].ID;
				}
			}
		
	}
}

__global__ void calcDiametersKernel(Cluster* clusters, int K, Point* points, int N, double* max)
{
	
	const int tid = blockIdx.x * MAX_NUM_OF_THREADS + threadIdx.x;
	int i;
	double distance;

	if (tid < N)
	{
		for (i = tid + 1; i < N; i++) // looking for the distances of the farthest 2 points inside a cluster
			if (points[tid].clusterID == points[i].clusterID)
			{
				distance = sqrt(pow(points[tid].x - points[i].x, 2) + pow(points[tid].y - points[i].y, 2));
				if (clusters[points[tid].clusterID].diameter < distance)
					clusters[points[tid].clusterID].diameter = distance;
					//max[i*K + points[i].clusterID] = distance;
				
					
			}
	}
}

__global__ void getMaxKernel(Cluster* clusters, int K, int N, double* max)
{
	const int tid = threadIdx.x;
	int i;
	double distance;

	for (i = 0; i < N; i++)
	{
		distance = max[i*K + tid];
		if (clusters[tid].diameter < distance)
			clusters[tid].diameter = distance;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t cudaRecalculatePoints(Point* points, int N, double dT)
{
	Point *dev_points = 0;
    cudaError_t cudaStatus;
	int blocks = N / MAX_NUM_OF_THREADS + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaRecalculatePoints - cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cudaFree(dev_points);
		return cudaStatus;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_points, N * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaRecalculatePoints - cudaMalloc failed!");
		cudaFree(dev_points);
		return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaRecalculatePoints - cudaMemcpy failed!");
		cudaFree(dev_points);
		return cudaStatus;
    }


    // Launch a kernel on the GPU with one thread for each element.
    moveKernel<<<blocks, MAX_NUM_OF_THREADS >>>(dev_points, N, dT);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_points);
		return cudaStatus;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaRecalculatePoints - cudaDeviceSynchronize returned error code %d after launching moveKernel!\n", cudaStatus);
		cudaFree(dev_points);
		return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(points, dev_points, N * sizeof(Point), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaRecalculatePoints - cudaMemcpy failed!");
		cudaFree(dev_points);
		return cudaStatus;
    }


    cudaFree(dev_points);
    return cudaStatus;
}

cudaError_t cudaGroupPoints(Cluster* clusters, int K, Point* points, int N, char* flag)
{
	Cluster* dev_clusters = 0;
	Point *dev_points = 0;
	char* dev_flag = 0;
	cudaError_t cudaStatus;
	int blocks = N / MAX_NUM_OF_THREADS + 1;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_points, N * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMalloc points failed!");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusters, K * sizeof(Cluster));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMalloc clusters failed!");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_flag, sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMalloc flag failed!");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy input points failed!");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_clusters, clusters, K * sizeof(Cluster), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy input clusters failed!");
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_flag, flag, sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy input flag failed!");
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Launch a kernel on the GPU with one thread for each element.
	groupKernel<<<blocks, MAX_NUM_OF_THREADS >>>(dev_clusters, K, dev_points, N, dev_flag);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "groupKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching groupKernel!\n", cudaStatus);
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, N * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy output points failed!");
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(flag, dev_flag, sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy output flag failed!");
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaFree(dev_flag);
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	return cudaStatus;
}

cudaError_t cudaCalcDiameters(Cluster* clusters, int K, Point* points, int N)
{
	Cluster* dev_clusters = 0;
	Point *dev_points = 0;
	double *dev_max = 0;
	//double *max = (double*)calloc(N*K, sizeof(double));
	cudaError_t cudaStatus;
	int blocks = N / MAX_NUM_OF_THREADS + 1, i;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCalcDiameters - cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_points, N * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCalcDiameters - cudaMalloc points failed!");
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusters, K * sizeof(Cluster));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCalcDiameters - cudaMalloc clusters failed!");
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	/*cudaStatus = cudaMalloc((void**)&dev_max, K * N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCalcDiameters - cudaMalloc clusters failed!");
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}*/

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCalcDiameters - cudaMemcpy input points failed!");
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_clusters, clusters, K * sizeof(Cluster), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCalcDiameters - cudaMemcpy input clusters failed!");
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	/*cudaStatus = cudaMemcpy(dev_max, max, K * N * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCalcDiameters - cudaMemcpy input max failed!");
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}*/

	// Launch a kernel on the GPU with one thread for each element.
	calcDiametersKernel << <blocks, MAX_NUM_OF_THREADS >> >(dev_clusters, K, dev_points, N, dev_max);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcDiametersKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcDiametersKernel!\n", cudaStatus);
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	//getMaxKernel <<<1, K >>>(dev_clusters, K, N, dev_max);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getMaxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getMaxKernel!\n", cudaStatus);
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(clusters, dev_clusters, K * sizeof(Cluster), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCalcDiameters - cudaMemcpy output clusters failed!");
		cudaFree(dev_max);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}
	
	cudaFree(dev_max);
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	return cudaStatus;
}