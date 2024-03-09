#### 方法一：优先队列

首先，考虑该问题的简化版：给定 $n$ 个非递减的非负数序列 $a_0,  a_1, \cdots, a_{n-1}$，找出第 $k$ 个最小的子序列和。

当 $k = 1$ 时，空序列即为答案。

当 $k \gt 1$，令 $(t, i)$ 表示以 $a_i$ 为最后一个元素且和为 $t$ 的子序列，则问题转化为求解第 $k$ 个最小的 $(t, i)$。我们使用一个小根堆来保存 $(t, i)$，初始时堆中只有一个元素 $(a_0, 0)$。为了能按从小到大的顺序依次获得子序列，当我们从小根堆取出堆顶元素 $(t, i)$ 时，我们需要进行以下操作：

- 将 $a_{i + 1}$ 拼接到子序列 $(t, i)$ 后，得到新的子序列 $(t + a_{i + 1}, i + 1)$，并将它加入堆中。

- 将子序列 $(t, i)$ 中的 $a_i$ 替换成 $a_{i+1}$，得到新的子序列 $(t + a_{i + 1} - a_i, i + 1)$，并将它加入堆中。

那么第 $k - 1$ 次取出的堆顶元素对应第 $k$ 个最小的子序列。

> 这种做法可以保证：
> 
> 1. 不重复地取出所有的非空子序列
> 
> 2. 依次取出的非空子序列和是非递减的。

根据以上讨论，我们可以求解出非递减的非负数序列的第 $k$ 个最小的子序列和，而原问题给出的条件中，允许序列中有负数。记原序列的非负数和与负数和分别为 $\textit{total}$ 与 $\textit{total}_\textit{neg}$，我们先将原序列中的负数替换成它的绝对值，然后从小到大进行排序，最后求解第 $k$ 个最小的子序列和 $t_k$。

$t_k$ 对应一个子序列 $s_k$，我们将 $t_k$ 加上 $\textit{total}_\textit{neg}$，那么 $t_k + \textit{total}_\textit{neg}$ 也对应原序列的一个子序列 $s_k'$，该子序列 $s_k'$ 的正数元素与 $s_k$ 相同，负数部分由绝对值不存在于 $s_k$ 的负数元素组成。因为 $t_k + \textit{total}_\textit{neg}$ 是非递减的，所以 $s_k'$ 是原序列的第 $k$ 个最小的子序列，$t_k + \textit{total}_\textit{neg}$ 是原序列的第 $k$ 个最小的子序列和。

同时，如果第 $k$ 个最小的子序列和为 $t_k + \textit{total}_\textit{neg}$，并且所有序列元素和为 $\textit{total} + \textit{total}_\textit{neg}$，那么 $\textit{total} + \textit{total}_\textit{neg} - (t_k + \textit{total}_\textit{neg}) = \textit{total} - t_k$ 即为第 $k$ 个最大的子序列和。

> $\textit{total} - t_k$ 对应没有出现在第 $k$ 个最小的子序列中的元素和，也对应一个子序列，因为 $t_k$ 是非递减的，所以 $\textit{total} - t_k$ 是非递增的，因此 $\textit{total} - t_k$ 即为第 $k$ 个最大的子序列和。

```C++ [sol1-C++]
class Solution {
public:
    long long kSum(vector<int> &nums, int k) {
        int n = nums.size();
        long long total = 0;
        for (int &x : nums) {
            if (x >= 0) {
                total += x;
            } else {
                x = -x;
            }
        }
        sort(nums.begin(), nums.end());

        long long ret = 0;
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        pq.push({nums[0], 0});
        for (int j = 2; j <= k; j++) {
            auto [t, i] = pq.top();
            pq.pop();
            ret = t;
            if (i == n - 1) {
                continue;
            }
            pq.push({t + nums[i + 1], i + 1});
            pq.push({t - nums[i] + nums[i + 1], i + 1});
        }
        return total - ret;
    }
};
```

```Java [sol1-Java]
class Solution {
    public long kSum(int[] nums, int k) {
        int n = nums.length;
        long total = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] >= 0) {
                total += nums[i];
            } else {
                nums[i] = -nums[i];
            }
        }
        Arrays.sort(nums);

        long ret = 0;
        PriorityQueue<long[]> pq = new PriorityQueue<long[]>((a, b) -> Long.compare(a[0], b[0]));
        pq.offer(new long[]{nums[0], 0});
        for (int j = 2; j <= k; j++) {
            long[] arr = pq.poll();
            long t = arr[0];
            int i = (int) arr[1];
            ret = t;
            if (i == n - 1) {
                continue;
            }
            pq.offer(new long[]{t + nums[i + 1], i + 1});
            pq.offer(new long[]{t - nums[i] + nums[i + 1], i + 1});
        }
        return total - ret;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public long KSum(int[] nums, int k) {
        int n = nums.Length;
        long total = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] >= 0) {
                total += nums[i];
            } else {
                nums[i] = -nums[i];
            }
        }
        Array.Sort(nums);

        long ret = 0;
        PriorityQueue<Tuple<long, int>, long> pq = new PriorityQueue<Tuple<long, int>, long>();
        pq.Enqueue(new Tuple<long, int>(nums[0], 0), nums[0]);
        for (int j = 2; j <= k; j++) {
            Tuple<long, int> tuple = pq.Dequeue();
            long t = tuple.Item1;
            int i = tuple.Item2;
            ret = t;
            if (i == n - 1) {
                continue;
            }
            pq.Enqueue(new Tuple<long, int>(t + nums[i + 1], i + 1), t + nums[i + 1]);
            pq.Enqueue(new Tuple<long, int>(t - nums[i] + nums[i + 1], i + 1), t - nums[i] + nums[i + 1]);
        }
        return total - ret;
    }
}
```

```Go [sol1-Go]
type PriorityQueue [][2]int64

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i][0] < pq[j][0]
}

func (pq PriorityQueue) Len() int {
    return len(pq)
}

func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x any) {
    (*pq) = append(*pq, x.([2]int64))
}

func (pq *PriorityQueue) Pop() any {
    n := len(*pq)
    x := (*pq)[n - 1]
    (*pq) = (*pq)[:n-1]
    return x
}

func kSum(nums []int, k int) int64 {
    n, total := len(nums), int64(0)
    for i := range nums {
        if nums[i] >= 0 {
            total += int64(nums[i])
        } else {
            nums[i] = -nums[i]
        }
    }
    sort.Ints(nums)

    ret := int64(0)
    pq := PriorityQueue{
        [2]int64{int64(nums[0]), 0},
    }
    for j := 2; j <= k; j++ {
        t, i := pq[0][0], pq[0][1]
        heap.Pop(&pq)
        ret = t
        if i == int64(n - 1) {
            continue
        }
        heap.Push(&pq, [2]int64{t + int64(nums[i + 1]), i + 1})
        heap.Push(&pq, [2]int64{t - int64(nums[i] - nums[i + 1]), i + 1})
    }
    return total - ret
}
```

```Python [sol1-Python3]
class Solution:
    def kSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total = 0
        for i in range(n):
            if nums[i] >= 0:
                total += nums[i]
            else:
                nums[i] = -nums[i]
        nums.sort()

        ret = 0
        pq = [(nums[0], 0)]
        for j in range(2, k + 1):
            t, i = heappop(pq)
            ret = t
            if i == n - 1:
                continue
            heappush(pq, (t + nums[i + 1], i + 1))
            heappush(pq, (t - nums[i] + nums[i + 1], i + 1))
        return total - ret
```

```C [sol1-C]
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

static int compare(const void *a, const void *b) {
    return *(int *)a - *(int *)b;
}

long long kSum(int* nums, int numsSize, int k) {
    long long total = 0;
    for (int i = 0; i < numsSize; i++) {
        if (nums[i] >= 0) {
            total += nums[i];
        } else {
            nums[i] = -nums[i];
        }
    }
    qsort(nums, numsSize, sizeof(int), compare);
    long long ret = 0;
    PriorityQueue *pq = createPriorityQueue(k * 2, greater);
    Node node;
    node.first = nums[0];
    node.second = 0;
    Push(pq, &node);
    for (int j = 2; j <= k; j++) {
        Node *p = Pop(pq);
        long long t = p->first;
        int i = p->second;
        ret = t;
        if (i == numsSize - 1) {
            continue;
        }
        node.first = t + nums[i + 1];
        node.second = i + 1;
        Push(pq, &node);
        node.first = t - nums[i] + nums[i + 1];
        node.second = i + 1;
        Push(pq, &node);
    }
    FreePriorityQueue(pq);
    return total - ret;
}
```

```JavaScript [sol1-JavaScript]
var kSum = function(nums, k) {
    const n = nums.length;
    let total = 0;
    for (let i = 0; i < nums.length; i++) {
        const x = nums[i];
        if (x >= 0) {
            total += x;
        } else {
            nums[i] = -x;
        }
    }
    
    nums.sort((a, b) => a - b);
    let ret = 0;
    const pq = new MinPriorityQueue();
    pq.enqueue([nums[0], 0], nums[0]);
    for (let j = 2; j <= k; j++) {
        const [t, i] = pq.front().element;
        pq.dequeue();
        ret = t;
        if (i == n - 1) {
            continue;
        }
        pq.enqueue([t + nums[i + 1], i + 1], t + nums[i + 1]);
        pq.enqueue([t - nums[i] + nums[i + 1], i + 1], t - nums[i] + nums[i + 1]);
    }
    return total - ret;
};
```

```TypeScript [sol1-TypeScript]
function kSum(nums: number[], k: number): number {
    const n: number = nums.length;
    let total: number = 0;
    for (let i: number = 0; i < nums.length; i++) {
        const x: number = nums[i];
        if (x >= 0) {
            total += x;
        } else {
            nums[i] = -x;
        }
    }
    
    nums.sort((a, b) => a - b);
    let ret: number = 0;
    const pq = new MinPriorityQueue();
    pq.enqueue([nums[0], 0], nums[0]);
    for (let j: number = 2; j <= k; j++) {
        const [t, i]: [number, number] = pq.front().element;
        pq.dequeue();
        ret = t;
        if (i == n - 1) {
            continue;
        }
        pq.enqueue([t + nums[i + 1], i + 1], t + nums[i + 1]);
        pq.enqueue([t - nums[i] + nums[i + 1], i + 1], t - nums[i] + nums[i + 1]);
    }
    return total - ret;
};
```

```Rust [sol1-Rust]
use std::collections::BinaryHeap;
use std::cmp::Reverse;

impl Solution {
    pub fn k_sum(nums: Vec<i32>, k: i32) -> i64 {
        let mut nums_copy = nums.to_vec();
        let n = nums_copy.len();
        let mut total = 0;

        for x in &mut nums_copy {
            if *x >= 0 {
                total += *x as i64;
            } else {
                *x = -*x;
            }
        }
        
        nums_copy.sort();
        let mut ret = 0;
        let mut pq: BinaryHeap<(Reverse<i64>, usize)> = BinaryHeap::new();
        pq.push((Reverse(nums_copy[0] as i64), 0));
        for j in 2..=k {
            let (t, i) = pq.pop().unwrap();
            ret = t.0;
            if i == n - 1 {
                continue;
            }
            pq.push((Reverse(t.0 + nums_copy[i + 1] as i64), i + 1));
            pq.push((Reverse(t.0 - nums_copy[i] as i64 + nums_copy[i + 1] as i64), i + 1));
        }
        (total - ret) as i64
    }
}
```

**复杂度分析**

- 时间复杂度：$O(n \log n + k \log k)$，其中 $n$ 是数组的长度。排序需要 $O(n \log n)$，堆获取第 $k$ 个最小子序列和需要 $O(k \log k)$。

- 空间复杂度：$O(k + \log n)$。排序需要 $O(\log n)$ 的栈空间，堆需要 $O(k)$ 的空间。

#### 方法二：二分

对于问题：给定 $n$ 个非递减的非负数序列 $a_0,  a_1, \cdots, a_{n-1}$，找出第 $k$ 个最小的子序列和。除了使用堆，我们还可以使用二分算法来解决。记 $\textit{total}_2$ 为非负数序列的和，令 $\textit{left} = 0, \textit{right} = \textit{total}_2$，那么在区间 $[\textit{left}, \textit{right}]$ 进行二分搜索。

令当前搜索的值为 $\textit{mid} = \lfloor \frac{(\textit{left} + \textit{right})}{2} \rfloor$，令 $\textit{cnt}$ 为和小于等于 $\textit{mid}$ 的非空子序列的个数，求解 $\textit{cnt}$ 可以使用深度优先搜索，假设当前搜索到第 $i$ 个元素，前面已选中的元素和为 $t$：

- 如果 $\textit{cnt} \ge k - 1$ 或 $t + a_i \gt \textit{mid}$，那么说明后续的搜索不必要，直接返回。

- 否则， $t + a_i$ 即对应以 $a_i$ 为最后一个元素的子序列和，将 $\textit{cnt}$ 加一，然后同时对 $(i + 1, t + a_i)$ 和 $(i + 1, t)$ 进行搜索。

最后，如果 $\textit{cnt} \ge k - 1$，令 $\textit{right} = \textit{mid} - 1$，否则令 $\textit{left} = \textit{mid} + 1$。当 $\textit{left} \gt \textit{right}$ 时，$\textit{left}$ 即为第 $k$ 个最小的子序列和。后续问题的求解同方法一。

```C++ [sol2-C++]
class Solution {
public:
    long long kSum(vector<int> &nums, int k) {
        int n = nums.size();
        long long total = 0, total2 = 0;
        for (int &x : nums) {
            if (x >= 0) {
                total += x;
            } else {
                x = -x;
            }
            total2 += abs(x);
        }
        sort(nums.begin(), nums.end());

        function<void(int, long long, long long, int &)> dfs = [&](int i, long long t, long long limit, int &cnt) {
            if (i == n || cnt >= k - 1 || t + nums[i] > limit) {
                return;
            }
            cnt++;
            dfs(i + 1, t + nums[i], limit, cnt);
            dfs(i + 1, t, limit, cnt);
        };

        long long left = 0, right = total2;
        while (left <= right) {
            long long mid = (left + right) / 2;
            int cnt = 0;
            dfs(0, 0, mid, cnt);
            if (cnt >= k - 1) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        return total - left;
    }
};
```

```Java [sol2-Java]
class Solution {
    int cnt;

    public long kSum(int[] nums, int k) {
        int n = nums.length;
        long total = 0, total2 = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] >= 0) {
                total += nums[i];
            } else {
                nums[i] = -nums[i];
            }
            total2 += Math.abs(nums[i]);
        }
        Arrays.sort(nums);

        long left = 0, right = total2;
        while (left <= right) {
            long mid = (left + right) / 2;
            cnt = 0;
            dfs(nums, k, n, 0, 0, mid);
            if (cnt >= k - 1) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        return total - left;
    }

    public void dfs(int[] nums, int k, int n, int i, long t, long limit) {
        if (i == n || cnt >= k - 1 || t + nums[i] > limit) {
            return;
        }
        cnt++;
        dfs(nums, k, n, i + 1, t + nums[i], limit);
        dfs(nums, k, n, i + 1, t, limit);
    }
}
```

```C# [sol2-C#]
public class Solution {
    int cnt;

    public long KSum(int[] nums, int k) {
        int n = nums.Length;
        long total = 0, total2 = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] >= 0) {
                total += nums[i];
            } else {
                nums[i] = -nums[i];
            }
            total2 += Math.Abs(nums[i]);
        }
        Array.Sort(nums);

        long left = 0, right = total2;
        while (left <= right) {
            long mid = (left + right) / 2;
            cnt = 0;
            DFS(nums, k, n, 0, 0, mid);
            if (cnt >= k - 1) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        return total - left;
    }

    public void DFS(int[] nums, int k, int n, int i, long t, long limit) {
        if (i == n || cnt >= k - 1 || t + nums[i] > limit) {
            return;
        }
        cnt++;
        DFS(nums, k, n, i + 1, t + nums[i], limit);
        DFS(nums, k, n, i + 1, t, limit);
    }
}
```

```Go [sol2-Go]
func kSum(nums []int, k int) int64 {
    n := len(nums)
    total, total2 := int64(0), int64(0)
    for i := range nums {
        if nums[i] >= 0 {
            total += int64(nums[i])
        } else {
            nums[i] = -nums[i]
        }
        total2 += int64(nums[i])
    }
    sort.Ints(nums)
    var cnt int
    var dfs func(int, int64, int64)
    dfs = func(i int, t, limit int64) {
        if i == n || cnt >= k - 1 || t + int64(nums[i]) > limit {
            return
        }
        cnt++
        dfs(i + 1, t + int64(nums[i]), limit)
        dfs(i + 1, t, limit)
    }

    left, right := int64(0), total2
    for left <= right {
        mid := (left + right) / 2
        cnt = 0
        dfs(0, 0, mid)
        if cnt >= k - 1 {
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    return total - left
}
```

```Python [sol2-Python3]
class Solution:
    def kSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total, total2 = 0, 0
        for i in range(n):
            if nums[i] >= 0:
                total += nums[i]
            else:
                nums[i] = -nums[i]
            total2 += nums[i]
        nums.sort()

        cnt = 0
        def dfs(i: int, t: int, limit: int) -> int:
            nonlocal cnt
            if i == n or cnt >= k - 1 or t + nums[i] > limit:
                return
            cnt += 1
            dfs(i + 1, t + nums[i], limit)
            dfs(i + 1, t, limit)
        
        left, right = 0, total2
        while left <= right:
            mid = (left + right) // 2
            cnt = 0
            dfs(0, 0, mid)
            if cnt >= k - 1:
                right = mid - 1
            else:
                left = mid + 1
        return total - left
```

```C [sol2-C]
static int compare(const void *a, const void *b) {
    return *(int *)a - *(int *)b;
}

void dfs(int *nums, int k, int n, int i, long long t, long long limit, int *cnt) {
    if (i == n || *cnt >= k - 1 || t + nums[i] > limit) {
        return;
    }
    (*cnt)++;
    dfs(nums, k, n, i + 1, t + nums[i], limit, cnt);
    dfs(nums, k, n, i + 1, t, limit, cnt);
}

long long kSum(int* nums, int numsSize, int k) {
    int n = numsSize;
    long long total = 0, total2 = 0;
    for (int i = 0; i < n; i++) {
        if (nums[i] < 0) {
            nums[i] = -nums[i];
        } else {
            total += nums[i];
        }
        total2 += abs(nums[i]);
    }

    qsort(nums, n, sizeof(int), compare);
    long long left = 0, right = total2;
    int cnt = 0;
    while (left <= right) {
        long long mid = (left + right) / 2;
        cnt = 0;
        dfs(nums, k, n, 0, 0, mid, &cnt);
        if (cnt >= k - 1) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return total - left;
}
```

```JavaScript [sol2-JavaScript]
var kSum = function(nums, k) {
    let n = nums.length;
    let total = 0, total2 = 0;
    for (let i = 0; i < n; i++) {
        if (nums[i] < 0) {
            nums[i] = -nums[i];
        } else {
            total += nums[i];
        }
        total2 += Math.abs(nums[i]);
    }

    nums.sort((a, b) => a - b);
    const dfs = (i, t, limit) => {
        if (i === n || cnt >= k - 1 || t + nums[i] > limit) {
            return;
        }
        cnt++;
        dfs(i + 1, t + nums[i], limit, cnt);
        dfs(i + 1, t, limit, cnt);
    };

    let left = 0, right = total2;
    let cnt = 0;
    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        cnt = 0;
        dfs(0, 0, mid);
        if (cnt >= k - 1) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return total - left;
};
```

```TypeScript [sol2-TypeScript]
function kSum(nums: number[], k: number): number {
    let n = nums.length;
    let total = 0, total2 = 0;
    for (let i = 0; i < n; i++) {
        if (nums[i] < 0) {
            nums[i] = -nums[i];
        } else {
            total += nums[i];
        }
        total2 += Math.abs(nums[i]);
    }

    nums.sort((a, b) => a - b);
    const dfs = (i: number, t: number, limit: number): void => {
        if (i === n || cnt >= k - 1 || t + nums[i] > limit) {
            return;
        }
        cnt++;
        dfs(i + 1, t + nums[i], limit);
        dfs(i + 1, t, limit);
    };

    let left = 0, right = total2;
    let cnt = 0;
    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        cnt = 0;
        dfs(0, 0, mid);
        if (cnt >= k - 1) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    console.log(left);
    return total - left;
};
```

```Rust [sol2-Rust]
impl Solution {
    pub fn k_sum(nums: Vec<i32>, k: i32) -> i64 {
        let n = nums.len();
        let mut nums = nums.to_vec();
        let mut total: i64 = 0;
        let mut total2: i64 = 0;
        for i in 0..n {
            if nums[i] >= 0 {
                total += nums[i] as i64;
            } else {
                nums[i] = -nums[i];
            }
            total2 += nums[i] as i64;
        }
        nums.sort();

        let mut cnt = 0;
        fn dfs(i: usize, t: i64, limit: i64, nums: &Vec<i32>, cnt: &mut i32, k: i32) {
            if i == nums.len() || *cnt >= k - 1 || t + nums[i] as i64 > limit {
                return;
            }
            *cnt += 1;
            dfs(i + 1, t + nums[i] as i64, limit, nums, cnt, k);
            dfs(i + 1, t, limit, nums, cnt, k);
        }
        
        let mut left = 0;
        let mut right = total2;
        while left <= right {
            let mid = (left + right) / 2;
            cnt = 0;
            dfs(0, 0, mid, &nums, &mut cnt, k);
            if cnt >= k - 1 {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        total - left
    }
}
```

**复杂度分析**

- 时间复杂度：$O(n \log n + k \log S)$。其中 $n$ 为序列元素数目，$S$ 为序列元素绝对值和。排序需要 $O(n \log n)$，二分+搜索需要 $k \log S$。

- 空间复杂度：$O(\min(n, k))$。搜索栈空间最多为 $O(\min(n, k))$。