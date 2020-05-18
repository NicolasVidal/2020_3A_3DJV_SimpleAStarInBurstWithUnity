using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class AStarTestScript : MonoBehaviour
{
    public Camera targetCamera;
    public Transform targetTransform;
    public float targetSpeed;
    public float rotateSpeed;
    public Collider groundCollider;
    public Transform[] obstacles;


    public int gridSize;

    public int[][] grid;

    private NativeArray<float> costMatrix;
    private NativeArray<float> heuristicMatrix;
    private List<Vector3> wayPoints;
    private Quaternion targetRotation;

    private void Start()
    {
        wayPoints = new List<Vector3>();

        grid = new int[gridSize][];
        for (var i = 0; i < gridSize; i++)
        {
            grid[i] = new int[gridSize];
            for (var j = 0; j < gridSize; j++)
            {
                grid[i][j] = 0;
            }
        }

        foreach (var obstacle in obstacles)
        {
            var position = obstacle.position;
            var i = Mathf.RoundToInt(position.z);
            var j = Mathf.RoundToInt(position.x);
            grid[i][j] = 1;
        }

        var linesCount = grid.Length;
        var colsCount = grid[0].Length;
        var costMatrixWidth = linesCount * colsCount;

        costMatrix = new NativeArray<float>(costMatrixWidth * costMatrixWidth, Allocator.Persistent);
        heuristicMatrix = new NativeArray<float>(costMatrixWidth, Allocator.Persistent);

        for (var i = 0; i < linesCount; i++)
        {
            for (var j = 0; j < colsCount; j++)
            {
                var sourceIdx = i * colsCount + j;

                for (var i1 = 0; i1 < linesCount; i1++)
                {
                    for (var j1 = 0; j1 < colsCount; j1++)
                    {
                        var targetIdx = i1 * colsCount + j1;

                        if (grid[i][j] != 1 && grid[i1][j1] != 1 && (i == i1 && Mathf.Abs(j - j1) == 1 ||
                                                                     j == j1 && Mathf.Abs(i - i1) == 1))
                        {
                            costMatrix[sourceIdx * costMatrixWidth + targetIdx] = 1;
                        }
                        else
                        {
                            costMatrix[sourceIdx * costMatrixWidth + targetIdx] = float.MaxValue;
                        }
                    }
                }
            }
        }
    }

    private void Update()
    {
        if (Input.GetMouseButtonUp(0) &&
            groundCollider.Raycast(targetCamera.ScreenPointToRay(Input.mousePosition), out var hit, 1000f))
        {
            var linesCount = grid.Length;
            var colsCount = grid[0].Length;
            var costMatrixWidth = linesCount * colsCount;

            //Update Heuristic Matrix to target node
            for (var i = 0; i < linesCount; i++)
            {
                for (var j = 0; j < colsCount; j++)
                {
                    var sourceIdx = i * colsCount + j;

                    heuristicMatrix[sourceIdx] = Mathf.Abs(i - Mathf.RoundToInt(hit.point.z)) +
                                                 Mathf.Abs(j - Mathf.RoundToInt(hit.point.x));
                }
            }

            var position = targetTransform.position;
            var job = new AStarJob
            {
                CostMatrix = costMatrix,
                NodesCount = costMatrixWidth,
                HeuristicMatrix = heuristicMatrix,
                StartNodeId = PosToId(position),
                EndNodeId = PosToId(hit.point),
                BestCost = new NativeArray<float>(1, Allocator.TempJob),
                ExploredNodesCount = new NativeArray<int>(1, Allocator.TempJob),
                BestPath = new NativeList<int>(Allocator.TempJob)
            };

            var sw = new Stopwatch();
            sw.Start();
            var handler = job.Schedule();

            handler.Complete();
            sw.Stop();
            Debug.Log($"Job Execution in {sw.ElapsedMilliseconds}ms");
            Debug.Log($"Explored Nodes Count : {job.ExploredNodesCount[0]}");
            Debug.Log($"Best Path Cost : {job.BestCost[0]}");

            var sb = new StringBuilder();
            wayPoints.Clear();
            for (var n = 1; n < job.BestPath.Length; n++)
            {
                var node = job.BestPath[n];
                var nodePos = new Vector3(node % gridSize, 1f, Mathf.FloorToInt(node / (float) gridSize));
                sb.Append($"{node}, ");
                wayPoints.Add(nodePos);
            }

            Debug.Log($"Best Path : {string.Join(", ", job.BestPath.ToArray())}");

            job.ExploredNodesCount.Dispose();
            job.BestCost.Dispose();
            job.BestPath.Dispose();
        }

        // Show Debug Path
        var lastPoint = targetTransform.position;
        for (var i = 0; i < wayPoints.Count; i++)
        {
            var point = wayPoints[i];
            Debug.DrawLine(lastPoint, point, Color.green, 0.1f);

            lastPoint = point;
        }

        // Move Target
        if (wayPoints.Count > 0)
        {
            var point = wayPoints[0];
            var position = targetTransform.position;
            var vectorToTarget = (point - position);
            var distanceToTarget = vectorToTarget.magnitude;

            if (distanceToTarget <= Time.deltaTime * targetSpeed)
            {
                targetTransform.position = point;
                wayPoints.RemoveAt(0);
            }
            else
            {
                targetTransform.position += Time.deltaTime * targetSpeed * (vectorToTarget) / distanceToTarget;

                targetRotation = Quaternion.LookRotation(vectorToTarget, Vector3.up);
            }
        }

        targetTransform.rotation =
            Quaternion.RotateTowards(targetTransform.rotation, targetRotation, rotateSpeed - Time.deltaTime);
    }

    private int PosToId(Vector3 pos)
    {
        return Mathf.RoundToInt(pos.z) * gridSize + Mathf.RoundToInt(pos.x);
    }

    private void OnDestroy()
    {
        costMatrix.Dispose();
        heuristicMatrix.Dispose();
    }
}