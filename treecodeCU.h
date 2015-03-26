//
//  treecodeCU.h
//  TreeTraverse
//
//  Created by Thomas Dickerson on 3/15/15.
//  Copyright (c) 2015 StickFigure Graphic Productions. All rights reserved.
//

#ifndef TreeTraverse_treecodeCU_h
#define TreeTraverse_treecodeCU_h

#include "treedefs.h"

template<size_t DIM, typename Float, size_t threadCt, size_t PPG, size_t MAX_LEVELS, size_t MAX_STACK_ENTRIES, size_t INTERACTION_THRESHOLD, TraverseMode Mode>
void traverseTreeCUDA(size_t nGroups, GroupInfoArray<DIM, Float, PPG> groupInfo, size_t startDepth,
					  NodeArray<DIM, Float> treeLevels[MAX_LEVELS], size_t treeCounts[MAX_LEVELS],
					  size_t n, ParticleArray<DIM, Float> particles, InteractionTypeArray(DIM, Float, Mode) interactions, Float softening, Float theta, size_t blockCt);
#endif
