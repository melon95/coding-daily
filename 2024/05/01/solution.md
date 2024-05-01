#### 方法一：使用小根堆维护可以雇佣的工人

**思路与算法**

根据题目描述，我们需要雇佣代价最小，并且在代价相等时下标最小的工人，因此我们可以使用小根堆维护所有当前可以雇佣的工人，小根堆的每个元素是一个二元组 $(\textit{cost}, \textit{id})$，分别表示工人的代价和下标。

初始时，我们需要将数组 $\textit{costs}$ 中的前 $\textit{candidates}$ 和后 $\textit{candidates}$ 个工人放入小根堆中。需要注意的是，如果 $\textit{candidates} \times 2 \geq n$（其中 $n$ 是数组 $\textit{costs}$ 的长度），前 $\textit{candidates}$ 和后 $\textit{candidates}$ 个工人存在重复，等价于将所有工人都放入小根堆中。

随后我们用 $\textit{left}$ 和 $\textit{right}$ 分别记录前面和后面可以选择的工人编号的**边界**，它们的初始值分别为 $\textit{candidates} - 1$ 和 $n - \textit{candidates}$。这样一来，我们就可以进行 $k$ 次操作，每次操作从小根堆中取出当前最优的工人。如果它的下标 $\textit{id} \leq \textit{left}$，那么它属于前面的工人，我们需要将 $\textit{left}$ 增加 $1$，并将新的 $(\textit{costs}[\textit{left}], \textit{left})$ 放入小根堆中。同理，如果 $\textit{id} \geq \textit{right}$，那么需要将 $\textit{right}$ 减少 $1$，并将新的 $(\textit{costs}[\textit{right}], \textit{right})$ 放入小根堆中。

如果 $\textit{left} + 1 \geq \textit{right}$，说明我们已经可以选择任意的工人，此时再向小根堆中添加工人会导致重复，因此只有在 $\textit{left} + 1 < \textit{right}$ 时，才会移动 $\textit{left}$ 或 $\textit{right}$ 并添加工人。

**代码**

```C++ [sol1-C++]
class Solution {
public:
    long long totalCost(vector<int>& costs, int k, int candidates) {
        int n = costs.size();
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
        int left = candidates - 1, right = n - candidates;
        if (left + 1 < right) {
            for (int i = 0; i <= left; ++i) {
                q.emplace(costs[i], i);
            }
            for (int i = right; i < n; ++i) {
                q.emplace(costs[i], i);
            }
        }
        else {
            for (int i = 0; i < n; ++i) {
                q.emplace(costs[i], i);
            }
        }
        long long ans = 0;
        for (int _ = 0; _ < k; ++_) {
            auto [cost, id] = q.top();
            q.pop();
            ans += cost;
            if (left + 1 < right) {
                if (id <= left) {
                    ++left;
                    q.emplace(costs[left], left);
                }
                else {
                    --right;
                    q.emplace(costs[right], right);
                }
            }
        }
        return ans;
    }
};
```

```Java [sol1-Java]
class Solution {
    public long totalCost(int[] costs, int k, int candidates) {
        int n = costs.length;
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>((a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);
        int left = candidates - 1, right = n - candidates;
        if (left + 1 < right) {
            for (int i = 0; i <= left; ++i) {
                pq.offer(new int[]{costs[i], i});
            }
            for (int i = right; i < n; ++i) {
                pq.offer(new int[]{costs[i], i});
            }
        } else {
            for (int i = 0; i < n; ++i) {
                pq.offer(new int[]{costs[i], i});
            }
        }
        long ans = 0;
        for (int i = 0; i < k; ++i) {
            int[] arr = pq.poll();
            int cost = arr[0], id = arr[1];
            ans += cost;
            if (left + 1 < right) {
                if (id <= left) {
                    ++left;
                    pq.offer(new int[]{costs[left], left});
                } else {
                    --right;
                    pq.offer(new int[]{costs[right], right});
                }
            }
        }
        return ans;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public long TotalCost(int[] costs, int k, int candidates) {
        int n = costs.Length;
        PriorityQueue<Tuple<int, int>, long> pq = new PriorityQueue<Tuple<int, int>, long>();
        int left = candidates - 1, right = n - candidates;
        if (left + 1 < right) {
            for (int i = 0; i <= left; ++i) {
                pq.Enqueue(new Tuple<int, int>(costs[i], i), (long) costs[i] * n + i);
            }
            for (int i = right; i < n; ++i) {
                pq.Enqueue(new Tuple<int, int>(costs[i], i), (long) costs[i] * n + i);
            }
        } else {
            for (int i = 0; i < n; ++i) {
                pq.Enqueue(new Tuple<int, int>(costs[i], i), (long) costs[i] * n + i);
            }
        }
        long ans = 0;
        for (int i = 0; i < k; ++i) {
            Tuple<int, int> pair = pq.Dequeue();
            int cost = pair.Item1, id = pair.Item2;
            ans += cost;
            if (left + 1 < right) {
                if (id <= left) {
                    ++left;
                    pq.Enqueue(new Tuple<int, int>(costs[left], left), (long) costs[left] * n + left);
                } else {
                    --right;
                    pq.Enqueue(new Tuple<int, int>(costs[right], right), (long) costs[right] * n + right);
                }
            }
        }
        return ans;
    }
}
```

```Python [sol1-Python3]
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        n = len(costs)
        q = list()
        left, right = candidates - 1, n - candidates
        if left + 1 < right:
            for i in range(left + 1):
                heappush(q, (costs[i], i))
            for i in range(right, n):
                heappush(q, (costs[i], i))
        else:
            for i in range(n):
                heappush(q, (costs[i], i))
        
        ans = 0
        for _ in range(k):
            cost, idx = heappop(q)
            ans += cost
            if left + 1 < right:
                if idx <= left:
                    left += 1
                    heappush(q, (costs[left], left))
                else:
                    right -= 1
                    heappush(q, (costs[right], right))
        return ans
```

```C [sol1-C]
typedef struct {
    int first;
    int second;
} Node;

typedef bool (*cmp)(const void *, const void *);

typedef struct {
    Node *arr;
    int capacity;
    int queueSize;
    cmp compare;
} PriorityQueue;

Node *createNode(int x, int y) {
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
        return &obj->arr[0];
    }
}

void FreePriorityQueue(PriorityQueue *obj) {
    free(obj->arr);
    free(obj);
}

bool greater(const void *a, const void *b) {
    int cost1 = ((Node *)a)->first;
    int id1 = ((Node *)a)->second;
    int cost2 = ((Node *)b)->first;
    int id2 = ((Node *)b)->second;
    return cost1 > cost2 || (cost1 == cost2 && id1 > id2);
}

long long totalCost(int* costs, int costsSize, int k, int candidates) {
    int n = costsSize;
    PriorityQueue *q = createPriorityQueue(costsSize, greater);
    int left = candidates - 1, right = n - candidates;
    Node node;
    if (left + 1 < right) {
        for (int i = 0; i <= left; ++i) {
            node.first = costs[i];
            node.second = i;
            Push(q, &node);
        }
        for (int i = right; i < n; ++i) {
            node.first = costs[i];
            node.second = i;
            Push(q, &node);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            node.first = costs[i];
            node.second = i;
            Push(q, &node);
        }
    }

    long long ans = 0;
    for (int i = 0; i < k; ++i) {
        int cost = Top(q)->first;
        int id = Top(q)->second;
        Pop(q);
        ans += cost;
        if (left + 1 < right) {
            if (id <= left) {
                ++left;
                node.first = costs[left];
                node.second = left;
                Push(q, &node);
            } else {
                --right;
                node.first = costs[right];
                node.second = right;
                Push(q, &node);
            }
        }
    }
    FreePriorityQueue(q);
    return ans;
}
```

```Go [sol1-Go]
func totalCost(costs []int, k int, candidates int) int64 {
    n := len(costs)
	h := &Heap{}
	left, right := candidates - 1, n - candidates

	if left + 1 < right {
		for i := 0; i <= left; i++ {
			heap.Push(h, []int{costs[i], i})
		}
		for i := right; i < n; i++ {
			heap.Push(h, []int{costs[i], i})
		}
	} else {
		for i := 0; i < n; i++ {
			heap.Push(h, []int{costs[i], i})
		}
	}

	ans := int64(0)
	for i := 0; i < k; i++ {
		p := heap.Pop(h).([]int)
        cost, id := p[0], p[1]
		ans += int64(cost)
		if left + 1 < right {
			if id <= left {
                left++
				heap.Push(h, []int{costs[left], left})
			} else {
                right--
				heap.Push(h, []int{costs[right], right})
			}
		}
	}
	return ans
}

type Heap [][]int

func (h Heap) Len() int { 
    return len(h) 
}

func (h Heap) Less(i, j int) bool { 
    if (h[i][0] == h[j][0]) {
        return h[i][1] < h[j][1]
    }
    return h[i][0] < h[j][0]
}

func (h Heap) Swap(i, j int) { 
    h[i], h[j] = h[j], h[i] 
}

func (h *Heap) Push(x interface{}) { 
    *h = append(*h, x.([]int)) 
}

func (h *Heap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n - 1]
	*h = old[0 : n - 1]
	return x
}
```

```JavaScript [sol1-JavaScript]
var totalCost = function(costs, k, candidates) {
    const n = costs.length;
    const q = new PriorityQueue({
        compare: (e1, e2) => {
            if (e1[0] < e2[0] || (e1[0] == e2[0] && e1[1] < e2[1])) {
                return -1;
            }
            return 1;
        }
    });
    let left = candidates - 1, right = n - candidates;
    if (left + 1 < right) {
        for (let i = 0; i <= left; ++i) {
            q.enqueue([costs[i], i]);
        }
        for (let i = right; i < n; ++i) {
            q.enqueue([costs[i], i]);
        }
    } else {
        for (let i = 0; i < n; ++i) {
            q.enqueue([costs[i], i]);
        }
    }
    let ans = 0;
    for (let i = 0; i < k; ++i) {
        const [cost, id] = q.dequeue();
        ans += cost;
        if (left + 1 < right) {
            if (id <= left) {
                ++left;
                q.enqueue([costs[left], left]);
            } else {
                --right;
                q.enqueue([costs[right], right]);
            }
        }
    }
    return ans;
};
```

```TypeScript [sol1-TypeScript]
function totalCost(costs: number[], k: number, candidates: number): number {
    const n = costs.length;
    const q = new PriorityQueue({
        compare: (e1, e2) => {
            if (e1[0] < e2[0] || (e1[0] == e2[0] && e1[1] < e2[1])) {
                return -1;
            }
            return 1;
        }
    });
    let left = candidates - 1, right = n - candidates;
    if (left + 1 < right) {
        for (let i = 0; i <= left; ++i) {
            q.enqueue([costs[i], i]);
        }
        for (let i = right; i < n; ++i) {
            q.enqueue([costs[i], i]);
        }
    } else {
        for (let i = 0; i < n; ++i) {
            q.enqueue([costs[i], i]);
        }
    }
    let ans = 0;
    for (let i = 0; i < k; ++i) {
        const [cost, id] = q.dequeue();
        ans += cost;
        if (left + 1 < right) {
            if (id <= left) {
                ++left;
                q.enqueue([costs[left], left]);
            } else {
                --right;
                q.enqueue([costs[right], right]);
            }
        }
    }
    return ans;
};
```

```Rust [sol1-Rust]
use std::collections::BinaryHeap;
use std::cmp::Reverse;

impl Solution {
    pub fn total_cost(costs: Vec<i32>, k: i32, candidates: i32) -> i64 {
        let n = costs.len();
        let mut q: BinaryHeap<Reverse<(i32, i32)>> = BinaryHeap::new();
        let mut left = (candidates - 1) as usize;
        let mut right = (n as i32 - candidates) as usize;

        if left + 1 < right {
            for i in 0..=left {
                q.push(Reverse((costs[i], i as i32)));
            }
            for i in right..n {
                q.push(Reverse((costs[i], i as i32)));
            }
        } else {
            for i in 0..n {
                q.push(Reverse((costs[i], i as i32)));
            }
        }

        let mut ans = 0i64;
        for _ in 0..k {
            let (cost, id) = q.pop().unwrap().0;
            ans += cost as i64;
            if left + 1 < right {
                if id as usize <= left {
                    left += 1;
                    q.push(Reverse((costs[left], left as i32)));
                } else {
                    right -= 1;
                    q.push(Reverse((costs[right], right as i32)));
                }
            }
        }
        ans
    }
}
```

**复杂度分析**

- 时间复杂度：$O((\textit{candidate} + k) \cdot \log \textit{candidate})$。
    - 构造初始的小根堆需要的时间为 $O(\textit{candidate} \cdot \log \textit{candidate})$，这一步可以优化至 $O(\textit{candidate})$，即将所有初始可以雇佣的员工放入数组中，然后一次性建立小根堆。但由于本题中 $\textit{candidate}$ 和 $n$ 同阶，因此不使用此优化。

    - 雇佣每一位工人需要的时间为 $O(\log candidate)$，一共需要雇佣 $k$ 位工人，总时间复杂度为 $O(k \cdot \log candidate)$。

  可以发现时间复杂度与 $n$ 无关，这也是很合理的，考虑 $n$ 非常大但 $\textit{candidate}$ 和 $k$ 非常小的情况，我们并不需要遍历整个数组 $\textit{costs}$。

- 空间复杂度：$O(\textit{candidate})$，即为小根堆需要使用的空间。