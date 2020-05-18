using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

[BurstCompile]
public struct AStarJob : IJob
{
    [ReadOnly]
    public NativeArray<float> CostMatrix;

    [ReadOnly]
    public NativeArray<float> HeuristicMatrix;

    [WriteOnly]
    public NativeArray<float> BestCost;

    [WriteOnly]
    public NativeArray<int> ExploredNodesCount;

    [WriteOnly]
    public NativeList<int> BestPath;

    public int NodesCount;

    public int StartNodeId;

    public int EndNodeId;

    public void Execute()
    {
        var nodesLeftToExploreIds = new NativeList<int>(NodesCount, Allocator.Temp);
        var nodesLeftToExploreCosts = new NativeList<float>(NodesCount, Allocator.Temp);
        var nodesPreviousElementIds = new NativeArray<int>(NodesCount, Allocator.Temp);
        var exploredNodesCountTmp = 0;

        for (var i = 0; i < NodesCount; i++)
        {
            nodesLeftToExploreIds.Add(i);
            nodesLeftToExploreCosts.Add(float.MaxValue);
        }

        nodesLeftToExploreCosts[StartNodeId] = 0;

        while (true)
        {
            var chosenNodeIdx = -1;
            var chosenNodeCost = float.MaxValue;
            var chosenNodeCostWithHeuristic = float.MaxValue;

            for (var i = 0; i < nodesLeftToExploreCosts.Length; i++)
            {
                var candidateNodeCost = nodesLeftToExploreCosts[i];
                var candidateNodeCostWithHeuristic =
                    candidateNodeCost + HeuristicMatrix[nodesLeftToExploreIds[i]]; // Uncomment For A*
                // var candidateNodeCostWithHeuristic = candidateNodeCost; // UnComment For Djikstra

                if (candidateNodeCostWithHeuristic < chosenNodeCostWithHeuristic ||
                    Math.Abs(candidateNodeCostWithHeuristic - chosenNodeCostWithHeuristic) < 0.00001f &&
                    candidateNodeCost > chosenNodeCost)
                {
                    chosenNodeIdx = i;
                    chosenNodeCost = candidateNodeCost;
                    chosenNodeCostWithHeuristic = candidateNodeCostWithHeuristic;
                }
            }

            if (chosenNodeIdx == -1)
            {
                ExploredNodesCount[0] = exploredNodesCountTmp;
                BestCost[0] = chosenNodeCost;
                return;
            }

            var chosenNodeId = nodesLeftToExploreIds[chosenNodeIdx];

            if (chosenNodeId == EndNodeId)
            {
                ExploredNodesCount[0] = exploredNodesCountTmp;
                BestCost[0] = chosenNodeCost;

                var nodeId = EndNodeId;
                var inversedPathQueue = new NativeList<int>(Allocator.Temp);
                inversedPathQueue.Add(nodeId);
                while (nodeId != StartNodeId)
                {
                    nodeId = nodesPreviousElementIds[nodeId];
                    inversedPathQueue.Add(nodeId);
                }

                var pathLength = inversedPathQueue.Length;
                for (var n = 0; n < pathLength; n++)
                {
                    BestPath.Add(inversedPathQueue[pathLength - 1 - n]);
                }

                return;
            }

            for (var j = 0; j < nodesLeftToExploreCosts.Length; j++)
            {
                var targetId = nodesLeftToExploreIds[j];

                var transitionCost = CostMatrix[chosenNodeId * NodesCount + targetId];
                if (transitionCost >= float.MaxValue)
                {
                    continue;
                }

                var candidateCost = transitionCost + chosenNodeCost;
                if (candidateCost < nodesLeftToExploreCosts[j])
                {
                    nodesLeftToExploreCosts[j] = candidateCost;
                    nodesPreviousElementIds[nodesLeftToExploreIds[j]] = chosenNodeId;
                }
            }

            nodesLeftToExploreIds.RemoveAtSwapBack(chosenNodeIdx);
            nodesLeftToExploreCosts.RemoveAtSwapBack(chosenNodeIdx);
            exploredNodesCountTmp++;
        }
    }
}