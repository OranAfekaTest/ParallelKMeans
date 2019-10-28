#include "Header.h"



int main(int argc, char *argv[])
{
	int N, K, T, LIMIT;
	double dT, QM;
	double *initValues = (double*)malloc(6*sizeof(double));
	int *sendcounts, *displs, i;
	int pointsPerSlave,
		pointsInMaster;

	Point* points = 0, *procPoint;
	Cluster* clusters = 0;

	int  namelen, numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Get_processor_name(processor_name, &namelen);

	MPI_Status status;
	MPI_Datatype PointMPIType;
	MPI_Datatype ClusterMPIType;
	createPointMPIType(&PointMPIType);
	createClusterMPIType(&ClusterMPIType);

	if (myid == MASTER)
	{
		//writeInputFile(IN_FNAME);
		readPointsFromFile(&points, IN_FNAME, &N, &K, &T, &dT, &LIMIT, &QM);
		initValues[0] = N;
		initValues[1] = K;
		initValues[2] = T;
		initValues[3] = dT;
		initValues[4] = LIMIT;
		initValues[5] = QM ;
		printf("N: %d, K: %d, T: %d, dT: %.2lf, LIMIT: %d, QM: %.2lf \n", N, K, T, dT, LIMIT, QM);
		fflush(stdout);
	}
	
	MPI_Bcast(initValues, 6, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

	if (myid != MASTER)
	{
		N = initValues[0];
		K = initValues[1];
		T = initValues[2];
		dT = initValues[3];
		LIMIT = initValues[4];
		QM = initValues[5];
	}

	pointsPerSlave = N / numprocs;
	pointsInMaster = N - (numprocs-1) * pointsPerSlave;
	
	sendcounts = (int*)malloc(numprocs * sizeof(int));
	displs = (int*)malloc(numprocs * sizeof(int));
	

	for (i = 0; i < numprocs; i++)
	{
		if (i == 0)
		{
			sendcounts[i] = pointsInMaster;
			displs[i] = 0;
		}
		else
		{
			sendcounts[i] = pointsPerSlave;
			displs[i] = pointsInMaster + (i - 1)*pointsPerSlave;
		}
		
	}
	
	procPoint = (Point*)malloc(sendcounts[myid] * sizeof(Point));
	
	MPI_Scatterv(points, sendcounts, displs, PointMPIType, procPoint, sendcounts[myid], PointMPIType, MASTER, MPI_COMM_WORLD);
	
	//if(myid == MASTER)
		kMeans(clusters, K, points, N, T, dT, LIMIT, QM, myid, numprocs, procPoint, sendcounts, displs, PointMPIType, ClusterMPIType);
	//else
	//{
	//	char time = 0, flag = 1;
	//	double q;
	//	clusters = (Cluster*)malloc(K * sizeof(Cluster));

	//	while (time == 0)
	//	{

	//		MPI_Bcast(clusters, K, ClusterMPIType, MASTER, MPI_COMM_WORLD);
	//		printf("***proc: %d flag: %c reaced!!***\n", myid, flag+48);
	//		fflush(stdout);

	//		while (flag != 0)
	//		{
	//			flag = 0;
	//			cudaGroupPoints(clusters, K, procPoint, sendcounts[myid], &flag);
	//			MPI_Gatherv(procPoint, sendcounts[myid], PointMPIType, points, sendcounts, displs, PointMPIType, MASTER, MPI_COMM_WORLD);
	//			MPI_Send(&flag, 1, MPI_CHAR, MASTER, 0, MPI_COMM_WORLD);
	//			
	//			MPI_Bcast(clusters, K, ClusterMPIType, MASTER, MPI_COMM_WORLD);

	//			//MPI_Bcast(&flag, 1, MPI_CHAR, MASTER, MPI_COMM_WORLD);
	//			MPI_Recv(&flag, 1, MPI_CHAR, MASTER, 0, MPI_COMM_WORLD, &status);
	//			if (flag == 0)
	//			{
	//				printf("***proc: %d flag: %c reaced!!***\n", myid, flag + 48);
	//				fflush(stdout);
	//			}

	//		}
	//		
	//		cudaRecalculatePoints(procPoint, sendcounts[myid], dT);
	//		MPI_Bcast(&q, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	//		if (q <= QM)
	//			time = 1;
	//		
	//	}
	//}
	

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	free(clusters);
	free(points);
	MPI_Finalize();
	return 0;
}

void kMeans(Cluster* clusters, int K, Point* points, int N, int T, double dT, int LIMIT, double QM, int rank, int numprocs, Point* procPoint, int* sizes, int* offsets, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType)
{
	const int pointsInProc = sizes[rank];
	int i, lim;
	double t, q;
	char flag, slaveFlag = 0;

	MPI_Status status;

	for (t = 0; t < T; t += dT)
	{
		if (t != 0)//move points only after first iteration
		{
			cudaRecalculatePoints(procPoint, pointsInProc, dT);
			free(clusters);
			//recalculatePoints(clusters, points, N, dT);
		}

		if (rank == MASTER)
			initClusters(&clusters, K, procPoint);
		else
			clusters = (Cluster*)malloc(K * sizeof(Cluster));

		MPI_Bcast(clusters, K, ClusterMPIType, MASTER, MPI_COMM_WORLD);


		for (lim = 0; lim < LIMIT; lim++)
		{
			flag = 0;
			/*printf("***\n--- proc: %d has %d points ---\n***\n", rank, pointsInProc);
			fflush(stdout);*/
			cudaGroupPoints(clusters, K, procPoint, pointsInProc, &flag);
			MPI_Gatherv(procPoint, pointsInProc, PointMPIType, points, sizes, offsets, PointMPIType, MASTER, MPI_COMM_WORLD);

			if (rank == MASTER)
			{
				for (i = 1; i < numprocs; i++)
				{
					MPI_Recv(&slaveFlag, 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
					flag += slaveFlag;

				}
			}
			else
				MPI_Send(&flag, 1, MPI_CHAR, MASTER, 0, MPI_COMM_WORLD);

			if (rank == MASTER)
			{
				updateClusters(clusters, K, points, N);
				recalculateClusters(clusters, K);
			}
			MPI_Bcast(clusters, K, ClusterMPIType, MASTER, MPI_COMM_WORLD);

			MPI_Bcast(&flag, 1, MPI_CHAR, MASTER, MPI_COMM_WORLD);


			if (flag == 0) //check if any points switched clusters during the iteration
				lim = LIMIT;
		}

		if (rank == MASTER)
		{
			//cudaCalcDiameters(clusters, K, points, N);
			calcDiameters(clusters, K, points, N);
			q = evaluateQuality(clusters, K);

			printf("*** t = %.2lf ***\n", t);
			printClusters(clusters, K);
		}

		MPI_Bcast(&q, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		if (q <= QM)
		{
			if (rank == MASTER)
			{
				writeOutputFile(OUT_FNAME, clusters, K, t, q);
				printf("\n*** Quality Measure Reached ***\n");
			}
			t = T;
		}
	}

}

void recalculatePoints(Cluster* clusters, Point* points, int N, double dT)
{
	int i;
	for(i = 0; i < N; i++)
	{
		points[i].x += dT * points[i].Vx;
		points[i].y += dT * points[i].Vy;
		points[i].clusterID = 0;
	}
	free(clusters);
}

void initClusters(Cluster** clusters, int K, Point* points)
{
	int i;
	*clusters = (Cluster*)malloc(K * sizeof(Cluster));
#pragma omp for
	for (i = 0; i < K; i++)
	{
		(*clusters)[i].ID = i;
		(*clusters)[i].diameter = 0;
		(*clusters)[i].centerX = points[i].x;
		(*clusters)[i].centerY = points[i].y;
		(*clusters)[i].sumX = 0;
		(*clusters)[i].sumY = 0;
		(*clusters)[i].counter = 0;
	}
}

void groupPoints(Cluster* clusters, int K, Point* points, int N, char* flag)
{
	int Pi, Ci;
	double newDistance, oldDistance;

	for (Pi = 0; Pi < N; Pi++)
	{
		for (Ci = 0; Ci < K; Ci++)
		{
			
				oldDistance = sqrt(pow(points[Pi].x - clusters[points[Pi].clusterID].centerX, 2) + pow(points[Pi].y - clusters[points[Pi].clusterID].centerY, 2));
				newDistance = sqrt(pow(points[Pi].x - clusters[Ci].centerX, 2) + pow(points[Pi].y - clusters[Ci].centerY, 2));
				if (newDistance < oldDistance) //calculate distances of points from cluster centers and make switch if necessery
				{
					*flag = 1;
					points[Pi].clusterID = clusters[Ci].ID;
				}
			
		}
	}
}

void recalculateClusters(Cluster* clusters, int K)
{
	int i;
#pragma omp for
	for (i = 0; i < K; i++)
	{//caculate the average position of points in cluster to get the center
		clusters[i].centerX = clusters[i].sumX / clusters[i].counter;
		clusters[i].centerY = clusters[i].sumY / clusters[i].counter;
	}
}

void updateClusters(Cluster* clusters, int K, Point* points, int N) {
	int i;

#pragma omp for
	for (i = 0; i < K; i++)
	{
		clusters[i].sumX = 0;
		clusters[i].sumY = 0;
		clusters[i].counter = 0;
	}

	for (i = 0; i < N; i++)
	{
		clusters[points[i].clusterID].sumX += points[i].x;
		clusters[points[i].clusterID].sumY += points[i].y;
		clusters[points[i].clusterID].counter++;
	}

}


void calcDiameters(Cluster* clusters, int K, Point* points, int N)
{
	int i, j;
	double distance, diameter;
#pragma omp for private (diameter)
	for (i = 0; i < N; i++)
	{
		diameter = clusters[points[i].clusterID].diameter;
#pragma omp parallel for private (distance)
		for (j = i + 1; j < N; j++) // looking for the distances of the farthest 2 points inside a cluster
			if (points[i].clusterID == points[j].clusterID)
			{
				distance = sqrt(pow(points[i].x - points[j].x, 2) + pow(points[i].y - points[j].y, 2));
				if (diameter < distance)
					diameter = distance;
			}
		clusters[points[i].clusterID].diameter = diameter;
	}
}

double evaluateQuality(Cluster* clusters, int K)
{
	int i, j;
	double devidedDistances = 0, q;
#pragma omp for
	for (i = 0; i < K; i++)
#pragma omp parallel for reduction(+ : devidedDistances)
		for (j = 0; j < K; j++)
			if (i != j)
			{
				devidedDistances += clusters[i].diameter / sqrt(pow(clusters[i].centerX - clusters[j].centerX, 2) + pow(clusters[i].centerY - clusters[j].centerY, 2));
				
			}
	q = devidedDistances / (K * (K - 1));
	printf("q: %.2lf \n", q);
	return q;
}

void readPointsFromFile(Point** buffer, char* fileName, int* N, int* K, int* T, double* dT, int* LIMIT, double* QM)
{
	FILE *f = fopen(fileName, "r");
	int i;
	double x, y, Vx, Vy;
	int res;
	
	if (!f)
	{
		printf("File not found.");
		fflush(stdout);
		return;
	}
	
	res = fscanf_s(f, "%d", N); // number of points
	res = fscanf_s(f, "%d", K); //number of clusters
	res = fscanf_s(f, "%d", T); //maximum time limit
	res = fscanf_s(f, "%lf", dT); //time interval
	res = fscanf_s(f, "%d", LIMIT); //maximum iterations limit
	res = fscanf_s(f, "%lf", QM); //quality mesurement for stopping condition

	fflush(stdout);
	
	*buffer = (Point*)malloc(*N * sizeof(Point));

	for (i = 0; i < *N; i++)
	{
			res = fscanf(f, "%lf", &x);
			res = fscanf(f, "%lf", &y);
			res = fscanf(f, "%lf", &Vx);
			res = fscanf(f, "%lf", &Vy);

			(*buffer)[i].x = x;
			(*buffer)[i].y = y;
			(*buffer)[i].Vx = Vx;
			(*buffer)[i].Vy = Vy;
			(*buffer)[i].clusterID = 0;
	}
	fclose(f);
}

void writeOutputFile(char* fileName, Cluster* clusters, int K, double t, double q)
{
	FILE* f = fopen(fileName, "w");
	int i;
	if (f == NULL)
	{
		printf("Failed opening the file. Exiting!\n");
		fflush(stdout);
		return;
	}

	fprintf(f, "First occurrence at t = %.2lf with q = %.2lf \n", t, q);
	fprintf(f, "Centers of the clusters : \n");
	for (i = 0; i < K; i++)
	{
		fprintf(f, " %lf	%lf \n", clusters[i].centerX, clusters[i].centerY);
	}


	fclose(f);
}

void writeInputFile(char* fileName)
{
	FILE* f = fopen(fileName, "w");
	
	int i;
	double randX, randY, randVx, randVy;
	
	int N = 50000*2;
	int K = 4;
	int T = 300;
	double dT = 0.1;
	int LIMIT = 2000;
	double QM = 0.3;
	
	if (f == NULL)
	{
		printf("Failed opening the file. Exiting!\n");
		fflush(stdout);
		return;
	}

	fprintf(f, "%d %d %d %lf %d %lf \n",  N, K, T, dT, LIMIT, QM);

	srand(time(NULL));
	for (i = 0; i < N; i++)
	{
		randX = (double)rand() / RAND_MAX * 200.0 - 100.0;
		randY = (double)rand() / RAND_MAX * 200.0 - 100.0;
		randVx = (double)rand() / RAND_MAX * 20.0 - 10.0;
		randVy = (double)rand() / RAND_MAX * 20.0 - 10.0;
		fprintf(f, " %lf %lf %lf %lf \n", randX, randY, randVx, randVy);
	}

	fclose(f);
}

void printClusters(Cluster *clusters, int K)
{
	int i;
	for (i = 0; i < K; i++)
	{
		printf("\n Cluster: %d; diameter: %.2lf; centerX: %.2lf; centerY: %.2lf; counter: %d \n", clusters[i].ID, clusters[i].diameter, clusters[i].centerX, clusters[i].centerY, clusters[i].counter);
		
	}
	printf("---------\n");
	fflush(stdout);
}

void createPointMPIType(MPI_Datatype* PointMPIType)
{
	MPI_Datatype type[5] = { MPI_DOUBLE,  MPI_DOUBLE,  MPI_DOUBLE,  MPI_DOUBLE, MPI_INT};
	int blocklen[5] = { 1, 1, 1, 1, 1 };
	MPI_Aint disp[5] = { offsetof(Point, x), offsetof(Point, y), offsetof(Point, Vx), offsetof(Point, Vy), offsetof(Point, clusterID) };

	MPI_Type_create_struct(5, blocklen, disp, type, PointMPIType);
	MPI_Type_commit(PointMPIType);
}

void createClusterMPIType(MPI_Datatype* ClusterMPIType)
{
	MPI_Datatype type[7] = { MPI_INT, MPI_DOUBLE,  MPI_DOUBLE,  MPI_DOUBLE,  MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	int blocklen[7] = { 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[7] = { offsetof(Cluster, ID), offsetof(Cluster, centerX), offsetof(Cluster, centerY), offsetof(Cluster, sumX), offsetof(Cluster, sumY), offsetof(Cluster, counter) };

	MPI_Type_create_struct(7, blocklen, disp, type, ClusterMPIType);
	MPI_Type_commit(ClusterMPIType);
}
