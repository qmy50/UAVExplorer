#ifndef PRM_ASTAR_H
#define PRM_ASTAR_H

#include <queue>
#include <global_planner/PRMKDTree.h>
#include <plan_env/grid_map_new.h>

namespace PRM {
    bool inClose(std::shared_ptr<Node> n, const std::unordered_set<std::shared_ptr<Node>>& close);
    
    std::vector<std::shared_ptr<Node>> AStar(const std::shared_ptr<KDTree>& roadmap,
                                             const std::shared_ptr<Node>& start,
                                             const std::shared_ptr<Node>& goal,
                                             const std::shared_ptr<GridMap>& map);
}

#endif