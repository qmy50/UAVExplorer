#include "PRMAstar.h"
#include <algorithm>
#include <ros/ros.h>

namespace PRM{
	bool inClose(std::shared_ptr<Node> n, const std::unordered_set<std::shared_ptr<Node>>& close){
		std::unordered_set<std::shared_ptr<Node>>::const_iterator got = close.find(n);
		return not (got == close.end());
	}

	// 2D non-inflated line check: sample along segment at map resolution
	static bool isLineOccupied2D(const std::shared_ptr<GridMap>& map,
	                             const Eigen::Vector3d& p1, const Eigen::Vector3d& p2){
		double res = map->getRes();
		double dist = (Eigen::Vector2d(p2.x() - p1.x(), p2.y() - p1.y())).norm();
		int steps = std::ceil(dist / res);
		for (int i = 0; i <= steps; ++i) {
			double t = (steps == 0) ? 0.0 : (double)i / steps;
			double x = p1.x() + t * (p2.x() - p1.x());
			double y = p1.y() + t * (p2.y() - p1.y());
			if (map->is2DOccupied(x, y)) return true;
		}
		return false;
	}

	// 3D non-inflated line check: sample along segment at map resolution
	static bool isLineOccupied3D(const std::shared_ptr<GridMap>& map,
	                             const Eigen::Vector3d& p1, const Eigen::Vector3d& p2){
		double res = map->getRes();
		double dist = (p2 - p1).norm();
		int steps = std::ceil(dist / res);
		for (int i = 0; i <= steps; ++i) {
			double t = (steps == 0) ? 0.0 : (double)i / steps;
			Eigen::Vector3d pt = p1 + t * (p2 - p1);
			if (map->isKnownOccupied(pt)) return true;
		}
		return false;
	}

	std::vector<std::shared_ptr<Node>> AStar(const std::shared_ptr<KDTree>& roadmap,
										     const std::shared_ptr<Node>& start,
										     const std::shared_ptr<Node>& goal,
										     const std::shared_ptr<GridMap>& map){
		std::vector<std::shared_ptr<Node>> path;

		// open: priority queue
		std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, CompareNode> open;
		start->g = 0.0;
		open.push(start);

		// close: unordered set
		std::unordered_set<std::shared_ptr<Node>> close;

		// record all nodes involved in the path search (need to clear data afterwards)
		std::vector<std::shared_ptr<Node>> record;
		record.push_back(start);
		bool findPath = false;
		while (ros::ok()){
			if (inClose(goal, close)){
				findPath = true;
				break;
			}

			if (open.size() == 0){
				findPath = false;
				//cout << "[Astar]: No valid path." << endl;
				break;
			}

			std::shared_ptr<Node> currNode = open.top();
			open.pop();

			// remove duplicates in open
			if (inClose(currNode, close)){
				continue;
			}

			// insert it into close
			close.insert(currNode);

			// iterate through all adjcent node:
			for (std::shared_ptr<Node> neighborNode : currNode->adjNodes){
				// Node must be not in close
				if (not inClose(neighborNode, close)){
					bool edge_free = map->is2DMapReady()
					    ? !isLineOccupied2D(map, currNode->pos, neighborNode->pos)
					    : !isLineOccupied3D(map, currNode->pos, neighborNode->pos);
					if (edge_free){
						double cost = currNode->g + (currNode->pos - neighborNode->pos).norm();
						if (cost < neighborNode->g){
							neighborNode->g = cost;
							neighborNode->f = cost + (neighborNode->pos - goal->pos).norm();
							open.push(neighborNode);
							neighborNode->parent = currNode;
							record.push_back(neighborNode);
						}
					}
				}
			}
		}


		// backtracking
		std::shared_ptr<Node> ptr = goal;
		while (ptr != NULL and findPath){
			path.push_back(ptr);
			ptr = ptr->parent;
		}
		std::reverse(path.begin(), path.end());

		// clear search cache data
		for (std::shared_ptr<Node> n : record){
			n->g = std::numeric_limits<double>::infinity();
			n->f = std::numeric_limits<double>::infinity();
			n->parent = NULL;
		}
		return path;
	}
}
