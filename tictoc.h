//
//  tictoc.h
//  TreeTraverse
//
//  Created by Thomas Dickerson on 3/15/15.
//  Copyright (c) 2015 StickFigure Graphic Productions. All rights reserved.
//

#ifndef TreeTraverse_tictoc_h
#define TreeTraverse_tictoc_h

#include "cuda.h"
cudaEvent_t start, stop; int ticline = 0;
#define toc {if(ticline) {printf("Lines %d to %d - ",ticline,__LINE__); tocf(); ticline = 0;} else printf("Two tocs;\n");}
#define tic {if(ticline) {printf("Two tics, toc;\n"); toc;} ticline=__LINE__; ticf();}
void ticf(){
	cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
}

float tocf(){
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	float time; cudaEventElapsedTime(&time, start, stop); printf("Elapsed time: %f ms\n", time);
	cudaEventDestroy(start); cudaEventDestroy(stop); return time;
}

#endif
