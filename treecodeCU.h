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

template<our_size_t DIM, typename Float, our_size_t threadCt, our_size_t PPG, our_size_t MAX_LEVELS, our_size_t MAX_STACK_ENTRIES, our_size_t INTERACTION_THRESHOLD, TraverseMode Mode, bool spam = false>
void traverseTreeCUDA(our_size_t nGroups, GroupInfoArray<DIM, Float, PPG> groupInfo, our_size_t startDepth,
					  NodeArray<DIM, Float> treeLevels[MAX_LEVELS], our_size_t treeCounts[MAX_LEVELS],
					  our_size_t n, ParticleArray<DIM, Float> particles, InteractionTypeArray(DIM, Float, Mode) interactions, Float softening, Float theta, our_size_t blockCt);
#endif
