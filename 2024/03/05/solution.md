#### 方法一：优先队列实现的 Dijkstra 算法

**思路**

正数权值的连通图中，求两点之间的最短路，很容易想到经典的[「Dijkstra 算法」](https://oi-wiki.org/graph/shortest-path/#Dijkstra-%E7%AE%97%E6%B3%95)。但是本题不仅仅是求出最短路，还要求出最短路径的数目。我们将在「Dijkstra 算法」的基础上，进行一些变动，来求出最短路径的数目。

观察[优先队列实现的「Dijkstra 算法」](https://oi-wiki.org/graph/shortest-path/#%E5%AE%9E%E7%8E%B0_2)，有以下数据结构：
- $e$ 是邻接表，这道题中需要我们自己根据 $\textit{roads}$ 创建。
- $q$ 是优先队列，元素是路径长度和点的编号。不停往外抛出队列中的最短路径和点。如果这个点是未被确定最短路径的点，那么这次出队列的操作，就将确定源到这个点的最短路径。然后依次访问这个点相邻的点，判断从这个点到相邻的点的路径，是否能刷新源相邻点的最短路径，如果能，则将路径长度和相邻点放入队列。
- $\textit{dis}$ 用来记录源到各个点当前最短路径的长度。会在访问当前出队列点的相邻点的过程中被刷新。
- $\textit{vis}$用来记录哪些点的最短路径已经被确定。在这里略显多余，可以用当前出队列的路径长度和点的最短路径的比较来代替。

除此之外，我们还需要一个新的数组 $\textit{ways}$。$\textit{ways}[v]$ 就表示源到点 $i$ 最短的路径数目，且最短路径长度为 $\textit{dis}[v]$。 $\textit{ways}$ 的更新时机与 $\textit{dis}$ 相同。在访问当前点 $u$ 的各个相邻点 $v$ 时，
- 如果从点 $u$ 到点 $v$ 路径，能刷新 $\textit{dis}[v]$，则更新 $\textit{dis}[v]$，并将 $\textit{ways}[v]$ 更新为 $\textit{ways}[u]$，表示有多少条源到点 $u$ 的最短路径，就有多少条源到点 $v$ 的最短路径。
- 如果从点 $u$ 到点 $v$ 路径，与 $\textit{dis}[v]$ 相等。那么 $\textit{ways}[v]$ 的值要加上 $\textit{ways}[u]$，表示点 $u$ 到点 $v$ 路径贡献了另一部分源到点 $v$ 的最短路径。
- 如果从点 $u$ 到点 $v$ 路径，大于 $\textit{dis}[v]$。那么无需操作 $\textit{dis}[v]$。

除了这个变动，其他部分与优先队列实现的「Dijkstra 算法」完全相同。到优先队列为空后，返回 $\textit{ways}$ 最后一个元素即可。


**代码**

```Python [sol1-Python3]
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        e = [[] for _ in range(n)]
        for x, y, t in roads:
            e[x].append([y, t])
            e[y].append([x, t])
        dis = [0] + [inf] * (n - 1)
        ways = [1] + [0] * (n - 1)

        q = [[0, 0]]
        while q:
            t, u = heappop(q)
            if t > dis[u]:
                continue
            for v, w in e[u]:
                if t + w < dis[v]:
                    dis[v] = t + w
                    ways[v] = ways[u]
                    heappush(q, [t + w, v])
                elif t + w == dis[v]:
                    ways[v] = (ways[u] + ways[v]) % mod
        return ways[-1]
```

```Java [sol1-Java]
class Solution {
    public int countPaths(int n, int[][] roads) {
        int mod = 1000000007;
        List<int[]>[] e = new List[n];
        for (int i = 0; i < n; i++) {
            e[i] = new ArrayList<int[]>();
        }
        for (int[] road : roads) {
            int x = road[0], y = road[1], t = road[2];
            e[x].add(new int[]{y, t});
            e[y].add(new int[]{x, t});
        }
        long[] dis = new long[n];
        Arrays.fill(dis, Long.MAX_VALUE);
        int[] ways = new int[n];

        PriorityQueue<long[]> pq = new PriorityQueue<long[]>((a, b) -> Long.compare(a[0], b[0]));
        pq.offer(new long[]{0, 0});
        dis[0] = 0;
        ways[0] = 1;

        while (!pq.isEmpty()) {
            long[] arr = pq.poll();
            long t = arr[0];
            int u = (int) arr[1];
            if (t > dis[u]) {
                continue;
            }
            for (int[] next : e[u]) {
                int v = next[0], w = next[1];
                if (t + w < dis[v]) {
                    dis[v] = t + w;
                    ways[v] = ways[u];
                    pq.offer(new long[]{t + w, v});
                } else if (t + w == dis[v]) {
                    ways[v] = (ways[u] + ways[v]) % mod;
                }
            }
        }
        return ways[n - 1];
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int CountPaths(int n, int[][] roads) {
        int mod = 1000000007;
        IList<Tuple<int, int>>[] e = new IList<Tuple<int, int>>[n];
        for (int i = 0; i < n; i++) {
            e[i] = new List<Tuple<int, int>>();
        }
        foreach (int[] road in roads) {
            int x = road[0], y = road[1], t = road[2];
            e[x].Add(new Tuple<int, int>(y, t));
            e[y].Add(new Tuple<int, int>(x, t));
        }
        long[] dis = new long[n];
        Array.Fill(dis, long.MaxValue);
        int[] ways = new int[n];

        PriorityQueue<Tuple<long, int>, long> pq = new PriorityQueue<Tuple<long, int>, long>();
        pq.Enqueue(new Tuple<long, int>(0, 0), 0);
        dis[0] = 0;
        ways[0] = 1;

        while (pq.Count > 0) {
            Tuple<long, int> tuple = pq.Dequeue();
            long t = tuple.Item1;
            int u = tuple.Item2;
            if (t > dis[u]) {
                continue;
            }
            foreach (Tuple<int, int> next in e[u]) {
                int v = next.Item1, w = next.Item2;
                if (t + w < dis[v]) {
                    dis[v] = t + w;
                    ways[v] = ways[u];
                    pq.Enqueue(new Tuple<long, int>(t + w, v), t + w);
                } else if (t + w == dis[v]) {
                    ways[v] = (ways[u] + ways[v]) % mod;
                }
            }
        }
        return ways[n - 1];
    }
}
```

```C++ [sol1-C++]
class Solution {
public:
    using LL = long long;
    int countPaths(int n, vector<vector<int>>& roads) {
        const long long mod = 1e9 + 7;
        vector<vector<pair<int, int>>> e(n);
        for (const auto& road : roads) {
            int x = road[0], y = road[1], t = road[2];
            e[x].emplace_back(y, t);
            e[y].emplace_back(x, t);
        }
        vector<long long> dis(n, LLONG_MAX);
        vector<long long> ways(n);

        priority_queue<pair<LL, int>, vector<pair<LL, int>>, greater<pair<LL, int>>> q;
        q.emplace(0, 0);
        dis[0] = 0;
        ways[0] = 1;
        
        while (!q.empty()) {
            auto [t, u] = q.top();
            q.pop();
            if (t > dis[u]) {
                continue;
            }
            for (auto &[v, w] : e[u]) {
                if (t + w < dis[v]) {
                    dis[v] = t + w;
                    ways[v] = ways[u];
                    q.emplace(t + w, v);
                } else if (t + w == dis[v]) {
                    ways[v] = (ways[u] + ways[v]) % mod;
                }
            }
        }
        return ways[n - 1];
    }
};
```

```C [sol1-C]
typedef struct Edge {
    int to;
    int cost;
    struct Edge *next;
} Edge;

typedef struct {
    long long first;
    int second;
} Node;

typedef bool (*cmp)(const void *, const void *);

typedef long long LL;

typedef struct {
    Node *arr;
    int capacity;
    int queueSize;
    cmp compare;
} PriorityQueue;

Edge *createEdge(int to, int cost) {
    Edge *obj = (Edge *)malloc(sizeof(Edge));
    obj->to = to;
    obj->cost = cost;
    obj->next = NULL;
    return obj;
}

void freeEdgeList(Edge *list) {
    while (list) {
        Edge *p = list;
        list = list->next;
        free(p);
    }
}

Node *createNode(long long x, int y) {
    Node *obj = (Node *)malloc(sizeof(Node));
    obj->first = x;
    obj->second = y;
    return obj;
}

PriorityQueue *createPriorityQueue(int size, cmp compare) {
    PriorityQueue *obj = (PriorityQueue *)malloc(sizeof(PriorityQueue));
    obj->arr = (Node *)malloc(sizeof(Node) * size);
    obj->queueSize = 0;
    obj->capacity = size;
    obj->compare = compare;
    return obj;
}

static void swap(Node *arr, int i, int j) {
    Node tmp;
    memcpy(&tmp, &arr[i], sizeof(Node));
    memcpy(&arr[i], &arr[j], sizeof(Node));
    memcpy(&arr[j], &tmp, sizeof(Node));
}

static void down(Node *arr, int size, int i, cmp compare) {
    for (int k = 2 * i + 1; k < size; k = 2 * k + 1) {
        // 父节点 (k - 1) / 2，左子节点 k，右子节点 k + 1
        if (k + 1 < size && compare(&arr[k], &arr[k + 1])) {
            k++;
        }
        if (compare(&arr[k], &arr[(k - 1) / 2])) {
            break;
        }
        swap(arr, k, (k - 1) / 2);
    }
}

void Heapfiy(PriorityQueue *obj) {
    for (int i = obj->queueSize / 2 - 1; i >= 0; i--) {
        down(obj->arr, obj->queueSize, i, obj->compare);
    }
}

void Push(PriorityQueue *obj, Node *node) {
    memcpy(&obj->arr[obj->queueSize], node, sizeof(Node));
    for (int i = obj->queueSize; i > 0 && obj->compare(&obj->arr[(i - 1) / 2], &obj->arr[i]); i = (i - 1) / 2) {
        swap(obj->arr, i, (i - 1) / 2);
    }
    obj->queueSize++;
}

Node* Pop(PriorityQueue *obj) {
    swap(obj->arr, 0, obj->queueSize - 1);
    down(obj->arr, obj->queueSize - 1, 0, obj->compare);
    Node *ret =  &obj->arr[obj->queueSize - 1];
    obj->queueSize--;
    return ret;
}

bool isEmpty(PriorityQueue *obj) {
    return obj->queueSize == 0;
}

Node* Top(PriorityQueue *obj) {
    if (obj->queueSize == 0) {
        return NULL;
    } else {
        return &obj->arr[obj->queueSize];
    }
}

void FreePriorityQueue(PriorityQueue *obj) {
    free(obj->arr);
    free(obj);
}

bool greater(const void *a, const void *b) {
   return ((Node *)a)->first > ((Node *)b)->first;
}

int countPaths(int n, int** roads, int roadsSize, int* roadsColSize) {
    const LL mod = 1e9 + 7;
    Edge *e[n];
    for (int i = 0; i < n; i++) {
        e[i] = NULL;
    }
    for (int i = 0; i < roadsSize; i++) {
        int x = roads[i][0], y = roads[i][1], t = roads[i][2];
        Edge *ex = createEdge(x, t);
        ex->next = e[y];
        e[y] = ex;
        Edge *ey = createEdge(y, t);
        ey->next = e[x];
        e[x] = ey;
    }
    
    LL dis[n], ways[n];
    for (int i = 0; i < n; i++) {
        dis[i] = LLONG_MAX;
        ways[i] = 0;
    }

    PriorityQueue *q = createPriorityQueue(n, greater);
    Node node;
    node.first = 0;
    node.second = 0;
    Push(q, &node);
    dis[0] = 0;
    ways[0] = 1;
    
    while (!isEmpty(q)) {
        Node *p = Pop(q);
        LL t = p->first;
        int u = p->second;
        if (t > dis[u]) {
            continue;
        }
        for (Edge *pEntry = e[u]; pEntry != NULL; pEntry = pEntry->next) {
            int v = pEntry->to, w = pEntry->cost;
            if (t + w < dis[v]) {
                dis[v] = t + w;
                ways[v] = ways[u];
                Node node;
                node.first = t + w;
                node.second = v;
                Push(q, &node);
            } else if (t + w == dis[v]) {
                ways[v] = (ways[u] + ways[v]) % mod;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        freeEdgeList(e[i]);
    }
    return ways[n - 1];
}
```

```Go [sol1-Go]
func countPaths(n int, roads [][]int) int {
    const mod = int64(1e9 + 7)
	e := make([][]Edge, n)
	for _, road := range roads {
		x, y, t := road[0], road[1], road[2]
		e[x] = append(e[x], Edge{y, t})
		e[y] = append(e[y], Edge{x, t})
	}

	dis := make([]int64, n)
	for i := range dis {
		dis[i] = math.MaxInt64
	}
	ways := make([]int64, n)

	q := PriorityQueue{{0, 0}}
	heap.Init(&q)
	dis[0] = 0
	ways[0] = 1

	for len(q) > 0 {
		p := heap.Pop(&q).(Pair)
		t, u := p.first, p.second
		if t > dis[u] {
			continue
		}
		for _, edge := range e[u] {
			v, w := edge.to, edge.cost
			if t + int64(w) < dis[v] {
				dis[v] = t + int64(w)
				ways[v] = ways[u]
				heap.Push(&q, Pair{t + int64(w), v})
			} else if t + int64(w) == dis[v] {
				ways[v] = (ways[u] + ways[v]) % mod
			}
		}
	}
	return int(ways[n - 1])
}

type Edge struct {
    to   int
    cost int
}

type Pair struct {
    first int64
    second int
}

type PriorityQueue []Pair

func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
}

func (pq PriorityQueue) Len() int {
    return len(pq)
}

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].first < pq[j].first
}

func (pq *PriorityQueue) Push(x any) {
    *pq = append(*pq, x.(Pair))
}

func (pq *PriorityQueue) Pop() any {
    n := len(*pq)
    x := (*pq)[n - 1]
    *pq = (*pq)[:n-1]
    return x
}
```

```JavaScript [sol1-JavaScript]
var countPaths = function(n, roads) {
    const mod = 1e9 + 7;
    const e = new Array(n).fill(0).map(() => new Array());
    for (const [x, y, t] of roads) {
        e[x].push([y, t]);
        e[y].push([x, t]);
    }

    const dis = [0].concat(Array(n - 1).fill(Infinity));
    const ways = [1].concat(Array(n - 1).fill(0));
    const q = new MinPriorityQueue();
    q.enqueue([0, 0], 0);
    
    while (!q.isEmpty()) {
        let t = q.front().element[0], u = q.front().element[1];
        q.dequeue();
        if (t > dis[u])
            continue;
        for (const [v, w] of e[u]) {
            if (t + w < dis[v]) {
                dis[v] = t + w;
                ways[v] = ways[u];
                q.enqueue([t + w, v], t + w);
            } else if (t + w == dis[v]) {
                ways[v] = (ways[u] + ways[v]) % mod;
            }
        }
    }
    return ways[n - 1];
};
```

```TypeScript [sol1-TypeScript]
function countPaths(n: number, roads: number[][]): number {
    const mod = 1e9 + 7;
    const e: number[][][] = new Array(n).fill(0).map(() => new Array());
    for (const [x, y, t] of roads) {
        e[x].push([y, t]);
        e[y].push([x, t]);
    }

    const dis: number[] = [0].concat(Array(n - 1).fill(Infinity));
    const ways: number[] = [1].concat(Array(n - 1).fill(0));
    const q = new MinPriorityQueue();
    q.enqueue([0, 0], 0);

    while (!q.isEmpty()) {
        let [t, u] = q.front().element;
        q.dequeue();
        if (t > dis[u]) continue;
        for (const [v, w] of e[u]) {
            if (t + w < dis[v]) {
                dis[v] = t + w;
                ways[v] = ways[u];
                q.enqueue([t + w, v], t + w);
            } else if (t + w === dis[v]) {
                ways[v] = (ways[u] + ways[v]) % mod;
            }
        }
    }
    return ways[n - 1];
};
```

```Rust [sol1-Rust]
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Copy, Clone, Eq, PartialEq)]
struct Pair {
    cost: i64,
    node: usize,
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Solution {
    pub fn count_paths(n: i32, roads: Vec<Vec<i32>>) -> i32 {
        const MOD: i64 = 1_000_000_007;
        let mut e: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n as usize];
        for road in roads {
            let x = road[0] as usize;
            let y = road[1] as usize;
            let t = road[2];
            e[x].push((y, t));
            e[y].push((x, t));
        }
        let mut dis: Vec<i64> = vec![i64::MAX; n as usize];
        let mut ways: Vec<i64> = vec![0; n as usize];

        let mut q = BinaryHeap::new();
        q.push(Pair { cost: 0, node: 0 });
        dis[0] = 0;
        ways[0] = 1;

        while let Some(Pair { cost: t, node: u }) = q.pop() {
            if t > dis[u] {
                continue;
            }
            for &(v, w) in &e[u] {
                if t + (w as i64)< dis[v] {
                    dis[v] = t + (w as i64);
                    ways[v] = ways[u];
                    q.push(Pair { cost: t + (w as i64), node: v });
                } else if t + (w as i64) == dis[v] {
                    ways[v] = (ways[v] + ways[u]) % MOD;
                }
            }
        }
        ways[(n - 1) as usize] as i32
    }
}
```

**复杂度分析**

- 时间复杂度：$O(m\times\log{m})$，其中 $m$ 是数组 $\textit{roads}$ 的长度。最小堆最多有 $O(m)$ 个元素。

- 时间复杂度：$O(m)$。邻接表和最小堆均占用 $O(m)$ 空间。