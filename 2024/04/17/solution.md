#### 方法一：深度优先搜索

考虑所有不在 $\textit{initial}$ 中的剩余节点，这些剩余节点组成一个图 $G$。

我们依次遍历 $\textit{initial}$ 中的节点，记当前遍历的节点为 $v$，将 $v$ 加入图 $G$（仅在此次遍历有效），同时使用深度优先搜索算法标记从节点 $v$ 可以访问到的节点集 $\textit{infectedSet}$；如果节点 $u \in \textit{infectedSet}$，那么我们将节点 $v$ 加入到 $\textit{infectedBy}[u]$ 中。

遍历结束后，我们使用 $\textit{count}[v]$ 统计图 $G$ 中只能被节点 $v$ 感染到的所有节点数目：如果 $\textit{infectedBy}[u]$ 的大小为 $1$，且 $\textit{infectedBy}[u][0] = v$，那么节点 $u$ 就是图 $G$ 中只能被节点 $v$ 感染到的节点。最后遍历 $\textit{count}$ 数组，取使 $\textit{count}[v]$ 最大且下标值最小的节点 $v$ 为答案。

```C++ [sol1-C++]
class Solution {
public:
    int minMalwareSpread(vector<vector<int>> &graph, vector<int> &initial) {
        int n = graph.size();
        vector<int> initialSet(n);
        for (int v : initial) {
            initialSet[v] = 1;
        }
        vector<vector<int>> infectedBy(n);
        for (int v : initial) {
            vector<int> infectedSet(n);
            dfs(graph, initialSet, infectedSet, v);
            for (int u = 0; u < n; u++) {
                if (infectedSet[u] == 1) {
                    infectedBy[u].push_back(v);
                }
            }
        }
        vector<int> count(n);
        for (int u = 0; u < n; u++) {
            if (infectedBy[u].size() == 1) {
                count[infectedBy[u][0]]++;
            }
        }
        int res = initial[0];
        for (int v : initial) {
            if (count[v] > count[res] || count[v] == count[res] && v < res) {
                res = v;
            }
        }
        return res;
    }

    void dfs(vector<vector<int>> &graph, vector<int> &initialSet, vector<int> &infectedSet, int v) {
        int n = graph.size();
        for (int u = 0; u < n; u++) {
            if (graph[v][u] == 0 || initialSet[u] == 1 || infectedSet[u] == 1) {
                continue;
            }
            infectedSet[u] = 1;
            dfs(graph, initialSet, infectedSet, u);
        }
    }
};
```

```Java [sol1-Java]
class Solution {
    public int minMalwareSpread(int[][] graph, int[] initial) {
        int n = graph.length;
        boolean[] initialSet = new boolean[n];
        for (int v : initial) {
            initialSet[v] = true;
        }
        List<Integer>[] infectedBy = new List[n];
        for (int i = 0; i < n; i++) {
            infectedBy[i] = new ArrayList<Integer>();
        }
        for (int v : initial) {
            boolean[] infectedSet = new boolean[n];
            dfs(graph, initialSet, infectedSet, v);
            for (int u = 0; u < n; u++) {
                if (infectedSet[u]) {
                    infectedBy[u].add(v);
                }
            }
        }
        int[] count = new int[n];
        for (int u = 0; u < n; u++) {
            if (infectedBy[u].size() == 1) {
                count[infectedBy[u].get(0)]++;
            }
        }
        int res = initial[0];
        for (int v : initial) {
            if (count[v] > count[res] || count[v] == count[res] && v < res) {
                res = v;
            }
        }
        return res;
    }

    public void dfs(int[][] graph, boolean[] initialSet, boolean[] infectedSet, int v) {
        int n = graph.length;
        for (int u = 0; u < n; u++) {
            if (graph[v][u] == 0 || initialSet[u] || infectedSet[u]) {
                continue;
            }
            infectedSet[u] = true;
            dfs(graph, initialSet, infectedSet, u);
        }
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int MinMalwareSpread(int[][] graph, int[] initial) {
        int n = graph.Length;
        bool[] initialSet = new bool[n];
        foreach (int v in initial) {
            initialSet[v] = true;
        }
        IList<int>[] infectedBy = new IList<int>[n];
        for (int i = 0; i < n; i++) {
            infectedBy[i] = new List<int>();
        }
        foreach (int v in initial) {
            bool[] infectedSet = new bool[n];
            DFS(graph, initialSet, infectedSet, v);
            for (int u = 0; u < n; u++) {
                if (infectedSet[u]) {
                    infectedBy[u].Add(v);
                }
            }
        }
        int[] count = new int[n];
        for (int u = 0; u < n; u++) {
            if (infectedBy[u].Count == 1) {
                count[infectedBy[u][0]]++;
            }
        }
        int res = initial[0];
        foreach (int v in initial) {
            if (count[v] > count[res] || count[v] == count[res] && v < res) {
                res = v;
            }
        }
        return res;
    }

    public void DFS(int[][] graph, bool[] initialSet, bool[] infectedSet, int v) {
        int n = graph.Length;
        for (int u = 0; u < n; u++) {
            if (graph[v][u] == 0 || initialSet[u] || infectedSet[u]) {
                continue;
            }
            infectedSet[u] = true;
            DFS(graph, initialSet, infectedSet, u);
        }
    }
}
```

```Go [sol1-Go]
func dfs(graph [][]int, initialSet, infectedSet []int, v int) {
    n := len(graph)
    for u := 0; u < n; u++ {
        if graph[v][u] == 0 || initialSet[u] == 1 || infectedSet[u] == 1 {
            continue
        }
        infectedSet[u] = 1
        dfs(graph, initialSet, infectedSet, u)
    }
}

func minMalwareSpread(graph [][]int, initial []int) int {
    n := len(graph)
    initialSet := make([]int, n)
    for _, v := range initial {
        initialSet[v] = 1
    }
    infectedBy := make([][]int, n)
    for _, v := range initial {
        infectedSet := make([]int, n)
        dfs(graph, initialSet, infectedSet, v)
        for u := 0; u < n; u++ {
            if infectedSet[u] == 1 {
                infectedBy[u] = append(infectedBy[u], v)
            }
        }
    }
    count := make([]int, n)
    for u := 0; u < n; u++ {
        if len(infectedBy[u]) == 1 {
            count[infectedBy[u][0]]++
        }
    }
    res := initial[0]
    for _, v := range initial {
        if count[v] > count[res] || count[v] == count[res] && v < res {
            res = v
        }
    }
    return res
}
```

```Python [sol1-Python3]
class Solution:
    def dfs(self, graph: List[List[int]], initialSet: List[int], infectedSet: List[int], v: int):
        n = len(graph)
        for u in range(n):
            if graph[v][u] == 0 or initialSet[u] == 1 or infectedSet[u] == 1:
                continue
            infectedSet[u] = 1
            self.dfs(graph, initialSet, infectedSet, u)
        return

    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        n = len(graph)
        initialSet = [0] * n
        for v in initial:
            initialSet[v] = 1
        infectedBy = [[] for _ in range(n)]
        for v in initial:
            infectedSet = [0] * n
            self.dfs(graph, initialSet, infectedSet, v)
            for u in range(n):
                if infectedSet[u] == 1:
                    infectedBy[u].append(v)
        count = [0] * n
        for u in range(n):
            if len(infectedBy[u]) == 1:
                count[infectedBy[u][0]] += 1
        res = initial[0]
        for v in initial:
            if count[v] > count[res] or (count[v] == count[res] and v < res):
                res = v
        return res
```

```C [sol1-C]
void dfs(int **graph, int graphSize, int *initialSet, int *infectedSet, int v) {
    int n = graphSize;
    for (int u = 0; u < n; u++) {
        if (graph[v][u] == 0 || initialSet[u] == 1 || infectedSet[u] == 1) {
            continue;
        }
        infectedSet[u] = 1;
        dfs(graph, graphSize, initialSet, infectedSet, u);
    }
}

int minMalwareSpread(int** graph, int graphSize, int* graphColSize, int* initial, int initialSize) {
    int n = graphSize;
    int initialSet[n];
    memset(initialSet, 0, sizeof(initialSet));
    for (int i = 0; i < initialSize; i++) {
        int v = initial[i];
        initialSet[v] = 1;
    }

    int infectedBy[n][n];
    int infectedBySize[n];
    memset(infectedBySize, 0, sizeof(infectedBySize));
    for (int i = 0; i < initialSize; i++) {
        int v = initial[i];
        int infectedSet[n];
        memset(infectedSet, 0, sizeof(infectedSet));
        dfs(graph, graphSize, initialSet, infectedSet, v);
        for (int u = 0; u < n; u++) {
            if (infectedSet[u] == 1) {
                infectedBy[u][infectedBySize[u]++] = v;
            }
        }
    }

    int count[n];
    memset(count, 0, sizeof(count));
    for (int u = 0; u < n; u++) {
        if (infectedBySize[u] == 1) {
            count[infectedBy[u][0]]++;
        }
    }
    int res = initial[0];
    for (int i = 0; i < initialSize; i++) {
        int v = initial[i];
        if (count[v] > count[res] || count[v] == count[res] && v < res) {
            res = v;
        }
    }
    return res;
}
```

```JavaScript [sol1-JavaScript]
var minMalwareSpread = function(graph, initial) {
    const n = graph.length;
    const initialSet = new Array(n).fill(0);
    for (const v of initial) {
        initialSet[v] = 1;
    }

    const infectedBy = new Array(n).fill(0).map(() => new Array());
    for (const v of initial) {
        const infectedSet = new Array(n).fill(0);
        dfs(graph, initialSet, infectedSet, v);
        for (let u = 0; u < n; u++) {
            if (infectedSet[u] == 1) {
                infectedBy[u].push(v);
            }
        }
    }

    const count = new Array(n).fill(0);
    for (let u = 0; u < n; u++) {
        if (infectedBy[u].length == 1) {
            count[infectedBy[u][0]]++;
        }
    }
    let res = initial[0];
    for (const v of initial) {
        if (count[v] > count[res] || count[v] == count[res] && v < res) {
            res = v;
        }
    }
    return res;
};

const dfs = (graph, initialSet, infectedSet, v) => {
    const n = graph.length;
    for (let u = 0; u < n; u++) {
        if (graph[v][u] == 0 || initialSet[u] == 1 || infectedSet[u] == 1) {
            continue;
        }
        infectedSet[u] = 1;
        dfs(graph, initialSet, infectedSet, u);
    }
}
```

```TypeScript [sol1-TypeScript]
function minMalwareSpread(graph: number[][], initial: number[]): number {
    const n = graph.length;
    const initialSet: number[] = new Array(n).fill(0);
    for (const v of initial) {
        initialSet[v] = 1;
    }

    const infectedBy: number[][] = new Array(n).fill(0).map(() => new Array());
    for (const v of initial) {
        const infectedSet: number[] = new Array(n).fill(0);
        dfs(graph, initialSet, infectedSet, v);
        for (let u = 0; u < n; u++) {
            if (infectedSet[u] == 1) {
                infectedBy[u].push(v);
            }
        }
    }

    const count: number[] = new Array(n).fill(0);
    for (let u = 0; u < n; u++) {
        if (infectedBy[u].length == 1) {
            count[infectedBy[u][0]]++;
        }
    }
    let res = initial[0];
    for (const v of initial) {
        if (count[v] > count[res] || count[v] == count[res] && v < res) {
            res = v;
        }
    }
    return res;
};

const dfs = (graph: number[][], initialSet: number[], infectedSet: number[], v: number) => {
    const n: number = graph.length;
    for (let u = 0; u < n; u++) {
        if (graph[v][u] == 0 || initialSet[u] == 1 || infectedSet[u] == 1) {
            continue;
        }
        infectedSet[u] = 1;
        dfs(graph, initialSet, infectedSet, u);
    }
}
```

```Rust [sol1-Rust]
use std::collections::HashSet;

impl Solution {
    fn dfs(graph: &Vec<Vec<i32>>, initial_set: &Vec<i32>, infected_set: &mut Vec<i32>, v: usize) {
        let n = graph.len() as i32;
        for u in 0..n as usize {
            if graph[v][u] == 0 || initial_set[u] == 1 || infected_set[u] == 1 {
                continue;
            }
            infected_set[u] = 1;
            Self::dfs(graph, initial_set, infected_set, u);
        }
    } 

    pub fn min_malware_spread(graph: Vec<Vec<i32>>, initial: Vec<i32>) -> i32 {
        let n = graph.len();
        let mut initial_set: Vec<i32> = vec![0; n];
        for &v in initial.iter() {
            initial_set[v as usize] = 1;
        }

        let mut infected_by: Vec<Vec<i32>> = vec![Vec::new(); n];
        for &v in initial.iter() {
            let mut infected_set: Vec<i32> = vec![0; n];
            Self::dfs(&graph, &initial_set, &mut infected_set, v as usize);
            for u in 0..n {
                if infected_set[u] == 1 {
                    infected_by[u].push(v);
                }
            }
        }

        let mut count: Vec<i32> = vec![0; n];
        for u in 0..n {
            if infected_by[u].len() == 1 {
                count[infected_by[u][0] as usize] += 1;
            }
        }
        let mut res = initial[0];
        for &v in initial.iter() {
            if count[v as usize] > count[res as usize] || 
               count[v as usize] == count[res as usize] && v < res {
                res = v;
            }
        }
        return res;
    }
}
```

**复杂度分析**

+ 时间复杂度：$O(n^3)$，其中 $n$ 为节点数目。最坏情况下，$n$ 个节点组成一个全连通图，那么深度优先搜索需要 $O(n^2)$，加上 $\textit{initial}$ 的外层循环，需要 $O(n^3)$。

+ 空间复杂度：$O(n^2)$。

#### 方法二：并查集

方法一中使用深度优先搜索算法来标记从节点 $v$ 可以访问到的节点集 $\textit{infectedSet}$，我们可以使用并查集来优化。使用图 $G$ 的所有节点初始化并查集 $\textit{uf}$。在遍历 $\textit{initial}$ 中的节点时，记当前遍历的节点为 $v$，我们只需要遍历所有图 $G$ 中与 $v$ 连接的节点，并将节点对应的子集加入到 $\textit{infectedSet}$ 中；对于子集 $u \in \textit{infectedSet}$，那么我们将节点 $v$ 加入到 $\textit{infectedBy}[u]$ 中。

遍历结束后，我们使用 $\textit{count}[v]$ 统计图 $G$ 中只能被节点 $v$ 感染到的所有节点数目：如果 $\textit{infectedBy}[u]$ 的大小为 $1$，且 $\textit{infectedBy}[u][0] = v$，那么子集 $u$ 的所有节点都只能被节点 $v$ 感染到； 我们遍历所有节点，如果节点 $w$ 在子集 $u$ 中，那么该节点就是图 $G$ 中只能被节点 $v$ 感染到的节点。最后遍历 $\textit{count}$ 数组，取使 $\textit{count}[v]$ 最大且下标值最小的节点 $v$ 为答案。

```C++ [sol2-C++]
class Solution {
public:
    int find(vector<int> &uf, int u) {
        if (uf[u] == u) {
            return u;
        }
        uf[u] = find(uf, uf[u]);
        return uf[u];
    }

    void merge(vector<int> &uf, int u, int v) {
        int ru = find(uf, u), rv = find(uf, v);
        uf[ru] = rv;
    }

    int minMalwareSpread(vector<vector<int>> &graph, vector<int> &initial) {
        int n = graph.size();
        vector<int> initialSet(n);
        for (int v : initial) {
            initialSet[v] = 1;
        }
        vector<int> uf(n);
        iota(uf.begin(), uf.end(), 0);
        for (int u = 0; u < n; u++) {
            if (initialSet[u] == 1) {
                continue;
            }
            for (int v = 0; v < n; v++) {
                if (initialSet[v] == 1) {
                    continue;
                }
                if (graph[u][v] == 1) {
                    merge(uf, u, v);
                }
            }
        }

        vector<vector<int>> infectedBy(n);
        for (int v : initial) {
            vector<int> infectedSet(n);
            for (int u = 0; u < n; u++) {
                if (initialSet[u] == 1 || graph[u][v] == 0) {
                    continue;
                }
                infectedSet[find(uf, u)] = 1;
            }
            for (int u = 0; u < n; u++) {
                if (infectedSet[u] == 1) {
                    infectedBy[u].push_back(v);
                }
            }
        }

        vector<int> count(n);
        for (int u = 0; u < n; u++) {
            if (infectedBy[u].size() != 1) {
                continue;
            }
            int v = infectedBy[u][0];
            for (int w = 0; w < n; w++) {
                if (find(uf, w) == find(uf, u)) {
                    count[v]++;
                }
            }
        }
        int res = initial[0];
        for (int v : initial) {
            if (count[v] > count[res] || count[v] == count[res] && v < res) {
                res = v;
            }
        }
        return res;
    }
};
```

```Java [sol2-Java]
class Solution {
    public int minMalwareSpread(int[][] graph, int[] initial) {
        int n = graph.length;
        boolean[] initialSet = new boolean[n];
        for (int v : initial) {
            initialSet[v] = true;
        }
        int[] uf = new int[n];
        for (int i = 0; i < n; i++) {
            uf[i] = i;
        }
        for (int u = 0; u < n; u++) {
            if (initialSet[u]) {
                continue;
            }
            for (int v = 0; v < n; v++) {
                if (initialSet[v]) {
                    continue;
                }
                if (graph[u][v] == 1) {
                    merge(uf, u, v);
                }
            }
        }

        List<Integer>[] infectedBy = new List[n];
        for (int i = 0; i < n; i++) {
            infectedBy[i] = new ArrayList<Integer>();
        }
        for (int v : initial) {
            boolean[] infectedSet = new boolean[n];
            for (int u = 0; u < n; u++) {
                if (initialSet[u] || graph[u][v] == 0) {
                    continue;
                }
                infectedSet[find(uf, u)] = true;
            }
            for (int u = 0; u < n; u++) {
                if (infectedSet[u]) {
                    infectedBy[u].add(v);
                }
            }
        }

        int[] count = new int[n];
        for (int u = 0; u < n; u++) {
            if (infectedBy[u].size() != 1) {
                continue;
            }
            int v = infectedBy[u].get(0);
            for (int w = 0; w < n; w++) {
                if (find(uf, w) == find(uf, u)) {
                    count[v]++;
                }
            }
        }
        int res = initial[0];
        for (int v : initial) {
            if (count[v] > count[res] || count[v] == count[res] && v < res) {
                res = v;
            }
        }
        return res;
    }

    public int find(int[] uf, int u) {
        if (uf[u] == u) {
            return u;
        }
        uf[u] = find(uf, uf[u]);
        return uf[u];
    }

    public void merge(int[] uf, int u, int v) {
        int ru = find(uf, u), rv = find(uf, v);
        uf[ru] = rv;
    }
}
```

```C# [sol2-C#]
public class Solution {
    public int MinMalwareSpread(int[][] graph, int[] initial) {
        int n = graph.Length;
        bool[] initialSet = new bool[n];
        foreach (int v in initial) {
            initialSet[v] = true;
        }
        int[] uf = new int[n];
        for (int i = 0; i < n; i++) {
            uf[i] = i;
        }
        for (int u = 0; u < n; u++) {
            if (initialSet[u]) {
                continue;
            }
            for (int v = 0; v < n; v++) {
                if (initialSet[v]) {
                    continue;
                }
                if (graph[u][v] == 1) {
                    Merge(uf, u, v);
                }
            }
        }

        IList<int>[] infectedBy = new IList<int>[n];
        for (int i = 0; i < n; i++) {
            infectedBy[i] = new List<int>();
        }
        foreach (int v in initial) {
            bool[] infectedSet = new bool[n];
            for (int u = 0; u < n; u++) {
                if (initialSet[u] || graph[u][v] == 0) {
                    continue;
                }
                infectedSet[Find(uf, u)] = true;
            }
            for (int u = 0; u < n; u++) {
                if (infectedSet[u]) {
                    infectedBy[u].Add(v);
                }
            }
        }

        int[] count = new int[n];
        for (int u = 0; u < n; u++) {
            if (infectedBy[u].Count != 1) {
                continue;
            }
            int v = infectedBy[u][0];
            for (int w = 0; w < n; w++) {
                if (Find(uf, w) == Find(uf, u)) {
                    count[v]++;
                }
            }
        }
        int res = initial[0];
        foreach (int v in initial) {
            if (count[v] > count[res] || count[v] == count[res] && v < res) {
                res = v;
            }
        }
        return res;
    }

    public int Find(int[] uf, int u) {
        if (uf[u] == u) {
            return u;
        }
        uf[u] = Find(uf, uf[u]);
        return uf[u];
    }

    public void Merge(int[] uf, int u, int v) {
        int ru = Find(uf, u), rv = Find(uf, v);
        uf[ru] = rv;
    }
}
```

```Go [sol2-Go]
func find(uf []int, u int) int {
    if uf[u] == u {
        return u
    }
    uf[u] = find(uf, uf[u])
    return uf[u]
}

func merge(uf []int, u, v int) {
    ru, rv := find(uf, u), find(uf, v)
    uf[ru] = rv
}

func minMalwareSpread(graph [][]int, initial []int) int {
    n := len(graph)
    initialSet := make([]int, n)
    for _, v := range initial {
        initialSet[v] = 1
    }
    uf := make([]int, n)
    for u := 0; u < n; u++ {
        uf[u] = u
    }
    for u := 0; u < n; u++ {
        if initialSet[u] == 1 {
            continue
        }
        for v := 0; v < n; v++ {
            if initialSet[v] == 1 {
                continue
            }
            if graph[u][v] == 1 {
                merge(uf, u, v)
            }
        }
    }
    infectedBy := make([][]int, n)
    for _, v := range initial {
        infectedSet := make([]int, n)
        for u := 0; u < n; u++ {
            if initialSet[u] == 1 || graph[u][v] == 0 {
                continue
            }
            infectedSet[find(uf, u)] = 1
        }
        for u := 0; u < n; u++ {
            if infectedSet[u] == 1 {
                infectedBy[u] = append(infectedBy[u], v)
            }
        }
    }
    count := make([]int, n)
    for u := 0; u < n; u++ {
        if len(infectedBy[u]) != 1 {
            continue
        }
        v := infectedBy[u][0]
        for w := 0; w < n; w++ {
            if find(uf, w) == find(uf, u) {
                count[v]++
            }
        }
    }
    res := initial[0]
    for _, v := range initial {
        if count[v] > count[res] || count[v] == count[res] && v < res {
            res = v
        }
    }
    return res
}
```

```Python [sol2-Python3]
class Solution:
    def find(self, uf: List[int], u: int) -> int:
        if uf[u] == u:
            return u
        uf[u] = self.find(uf, uf[u])
        return uf[u]

    def merge(self, uf: List[int], u: int, v: int):
        ru, rv = self.find(uf, u), self.find(uf, v)
        uf[ru] = rv

    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        n = len(graph)
        initialSet = [0] * n
        for v in initial:
            initialSet[v] = 1
        uf = [i for i in range(n)]
        for u in range(n):
            if initialSet[u] == 1:
                continue
            for v in range(n):
                if initialSet[v] == 1:
                    continue
                if graph[u][v] == 1:
                    self.merge(uf, u, v)
        infectedBy = [[] for _ in range(n)]
        for v in initial:
            infectedSet = [0] * n
            for u in range (n):
                if initialSet[u] == 1 or graph[u][v] == 0:
                    continue
                infectedSet[self.find(uf, u)] = 1
            for u in range(n):
                if infectedSet[u] == 1:
                    infectedBy[u].append(v)

        count = [0] * n
        for u in range(n):
            if len(infectedBy[u]) != 1:
                continue
            v = infectedBy[u][0]
            for w in range(n):
                if self.find(uf, w) == self.find(uf, u):
                    count[v] += 1
        res = initial[0]
        for v in initial:
            if count[v] > count[res] or (count[v] == count[res] and v < res):
                res = v
        return res
```

```C [sol2-C]
int find(int *uf, int u) {
    if (uf[u] == u) {
        return u;
    }
    uf[u] = find(uf, uf[u]);
    return uf[u];
}

void merge(int *uf, int u, int v) {
    int ru = find(uf, u), rv = find(uf, v);
    uf[ru] = rv;
}

int minMalwareSpread(int** graph, int graphSize, int* graphColSize, int* initial, int initialSize) {
    int n = graphSize;
    int initialSet[n];
    memset(initialSet, 0, sizeof(initialSet));
    for (int i = 0; i < initialSize; i++) {
        initialSet[initial[i]] = 1;
    }

    int uf[n];
    for (int i = 0; i < n; i++) {
        uf[i] = i;
    }
    for (int u = 0; u < n; u++) {
        if (initialSet[u] == 1) {
            continue;
        }
        for (int v = 0; v < n; v++) {
            if (initialSet[v] == 1) {
                continue;
            }
            if (graph[u][v] == 1) {
                merge(uf, u, v);
            }
        }
    }

    int infectedBy[n][n];
    int infectedBySize[n];
    memset(infectedBySize, 0, sizeof(infectedBySize));
    for (int i = 0; i < initialSize; i++) {
        int v = initial[i];
        int infectedSet[n];
        memset(infectedSet, 0, sizeof(infectedSet));
        for (int u = 0; u < n; u++) {
            if (initialSet[u] == 1 || graph[u][v] == 0) {
                continue;
            }
            infectedSet[find(uf, u)] = 1;
        }
        for (int u = 0; u < n; u++) {
            if (infectedSet[u] == 1) {
                infectedBy[u][infectedBySize[u]++] = v;
            }
        }
    }

    int count[n];
    memset(count, 0, sizeof(count));
    for (int u = 0; u < n; u++) {
        if (infectedBySize[u] != 1) {
            continue;
        }
        int v = infectedBy[u][0];
        for (int w = 0; w < n; w++) {
            if (find(uf, w) == find(uf, u)) {
                count[v]++;
            }
        }
    }
    int res = initial[0];
    for (int i = 0; i < initialSize; i++) {
        int v = initial[i];
        if (count[v] > count[res] || count[v] == count[res] && v < res) {
            res = v;
        }
    }
    return res;
}
```

```JavaScript [sol2-JavaScript]
function find(uf, u) {
    if (uf[u] === u) {
        return u;
    }
    uf[u] = find(uf, uf[u]);
    return uf[u];
}

function merge(uf, u, v) {
    const ru = find(uf, u);
    const rv = find(uf, v);
    if (ru !== rv) {
        uf[ru] = rv;
    }
}

var minMalwareSpread = function(graph, initial) {
    const n = graph.length;
    const initialSet = new Array(n).fill(0);
    for (const v of initial) {
        initialSet[v] = 1;
    }
    const uf = new Array(n).fill(0).map((_, i) => i);
    for (let u = 0; u < n; u++) {
        if (initialSet[u]) {
            continue;
        }
        for (let v = 0; v < n; v++) {
            if (initialSet[v]) {
                continue;
            }
            if (graph[u][v] === 1) {
                merge(uf, u, v);
            }
        }
    }

    const infectedBy = new Array(n).fill(0).map(() => []);
    for (const v of initial) {
        const infectedSet = new Array(n).fill(0);
        for (let u = 0; u < n; u++) {
            if (initialSet[u]) {
                continue;
            }
            if (graph[u][v] === 1) {
                infectedSet[find(uf, u)] = 1;
            }
        }
        for (let u = 0; u < n; u++) {
            if (infectedSet[u] === 1) {
                infectedBy[u].push(v);
            }
        }
    }

    const count = new Array(n).fill(0);
    for (let u = 0; u < n; u++) {
        if (infectedBy[u].length !== 1) {
            continue;
        }
        const v = infectedBy[u][0];
        for (let w = 0; w < n; w++) {
            if (find(uf, w) === find(uf, u)) {
                count[v]++;
            }
        }
    }
    let res = initial[0];
    for (const v of initial) {
        if (count[v] > count[res] || (count[v] === count[res] && v < res)) {
            res = v;
        }
    }
    return res;
};
```

```TypeScript [sol2-TypeScript]
function find(uf: number[], u: number): number {
    if (uf[u] === u) {
        return u;
    }
    uf[u] = find(uf, uf[u]);
    return uf[u];
}

function merge(uf: number[], u: number, v: number): void {
    const ru = find(uf, u);
    const rv = find(uf, v);
    if (ru !== rv) {
        uf[ru] = rv;
    }
}

function minMalwareSpread(graph: number[][], initial: number[]): number {
    const n = graph.length;
    const initialSet = new Array(n).fill(0);
    for (const v of initial) {
        initialSet[v] = 1;
    }
    const uf = new Array(n).fill(0).map((_, i) => i);
    for (let u = 0; u < n; u++) {
        if (initialSet[u]) {
            continue;
        }
        for (let v = 0; v < n; v++) {
            if (initialSet[v]) {
                continue;
            }
            if (graph[u][v] === 1) {
                merge(uf, u, v);
            }
        }
    }

    const infectedBy = new Array(n).fill(0).map(() => []);
    for (const v of initial) {
        const infectedSet = new Array(n).fill(0);
        for (let u = 0; u < n; u++) {
            if (initialSet[u]) {
                continue;
            }
            if (graph[u][v] === 1) {
                infectedSet[find(uf, u)] = 1;
            }
        }
        for (let u = 0; u < n; u++) {
            if (infectedSet[u] === 1) {
                infectedBy[u].push(v);
            }
        }
    }

    const count = new Array(n).fill(0);
    for (let u = 0; u < n; u++) {
        if (infectedBy[u].length !== 1) {
            continue;
        }
        const v = infectedBy[u][0];
        for (let w = 0; w < n; w++) {
            if (find(uf, w) === find(uf, u)) {
                count[v]++;
            }
        }
    }
    let res = initial[0];
    for (const v of initial) {
        if (count[v] > count[res] || (count[v] === count[res] && v < res)) {
            res = v;
        }
    }
    return res;
};
```

```Rust [sol2-Rust]
impl Solution {
    fn find(uf: &mut Vec<usize>, u: usize) -> usize {
        if uf[u] == u {
            u
        } else {
            uf[u] = Self::find(uf, uf[u]);
            uf[u]
        }
    }

    fn merge(uf: &mut Vec<usize>, u: usize, v: usize) {
        let ru = Self::find(uf, u);
        let rv = Self::find(uf, v);
        if ru != rv {
            uf[ru] = rv;
        }
    }

    pub fn min_malware_spread(graph: Vec<Vec<i32>>, initial: Vec<i32>) -> i32 {
        let n = graph.len();
        let mut initial_set: Vec<i32> = vec![0; n];
        for &v in initial.iter() {
            initial_set[v as usize] = 1;
        }
        let mut uf: Vec<usize> = vec![0; n];
        for i in 0..n {
            uf[i] = i;
        }
        for u in 0..n {
            if initial_set[u] == 1 {
                continue;
            }
            for v in 0..n {
                if initial_set[v] == 1 {
                    continue;
                }
                if graph[u][v] == 1 {
                    Self::merge(&mut uf, u, v);
                }
            }
        }

        let mut infected_by = vec![Vec::new(); n];
        for &v in initial.iter() {
            let mut infected_set = vec![0; n];
            for u in 0..n {
                if initial_set[u] == 1 {
                    continue;
                }
                if graph[u][v as usize] == 1 {
                    infected_set[Self::find(&mut uf, u)] = 1;
                }
            }
            for u in 0..n {
                if infected_set[u] == 1 {
                    infected_by[u].push(v);
                }
            }
        }

        let mut count: Vec<i32> = vec![0; n];
        for u in 0..n {
            if infected_by[u].len() != 1 {
                continue;
            }
            let v = infected_by[u][0];
            for w in 0..n {
                if (Self::find(&mut uf, w) == Self::find(&mut uf, u)) {
                    count[v as usize] += 1;
                }
            }
        }
        let mut res = initial[0];
        for &v in initial.iter() {
            if count[v as usize] > count[res as usize] || (count[v as usize] == count[res as usize] && v < res) {
                res = v;
            }
        }
        return res;
    }
}
```

**复杂度分析**

+ 时间复杂度：$O(n^2 \log n)$，其中 $n$ 为节点数目。并查集查找需要 $O(\log n)$（非按秩合并），总共有 $O(n^2)$ 次查找。

+ 空间复杂度：$O(n^2)$。