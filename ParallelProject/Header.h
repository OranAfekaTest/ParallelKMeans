#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define IN_FNAME "input.txt"
#define OUT_FNAME "output.txt"
#define MAX_NUM_OF_THREADS 1024
#define MASTER 0
#define SLAVE1 1
#define SLAVE2 2

struct Cluster
{
	int ID;
	double diameter;
	double centerX;
	double centerY;
	double sumX;
	double sumY;
	int counter;
}typedef Cluster;

struct Point
{
	double x;
	double y;
	double Vx;
	double Vy;
	int clusterID;
}typedef Point;

cudaError_t cudaCalcDiameters(Cluster* clusters, int K, Point* points, int N);
cudaError_t cudaGroupPoints(Cluster* clusters, int K, Point* points, int N, char* flag);
cudaError_t cudaRecalculatePoints(Point* points, int N, double dT);
void calcDiameters(Cluster* clusters, int K, Point* points, int N);
void recalculatePoints(Cluster* clusters, Point* points, int N, double dT);
void initClusters(Cluster** clusters, int K, Point* points);
void groupPoints(Cluster* clusters, int K, Point* points, int N, char* flag);
void recalculateClusters(Cluster* clusters, int K);
void kMeans(Cluster* clusters, int K, Point* points, int N, int T, double dT, int LIMIT, double QM, int rank, int numprocs, Point* procPoint, int* sizes, int* offsets, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType);
double evaluateQuality(Cluster* clusters, int K);
void readPointsFromFile(Point** buffer, char* fileName, int* N, int* K, int* T, double* dT, int* LIMIT, double* QM);
void writeInputFile(char* fileName);
void writeOutputFile(char* fileName, Cluster* clusters, int K, double t, double q);
void updateClusters(Cluster* clusters, int K, Point* points, int N);
void createClusterMPIType(MPI_Datatype* ClusterMPIType);
void createPointMPIType(MPI_Datatype* PointMPIType);
void printClusters(Cluster *clusters, int K);