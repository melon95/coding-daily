#### 方法一：双优先队列

将 $[0, i]$ 范围内的计数器初始所示数字操作成满足所有条件 $\textit{nums}[j] + 1 = \textit{nums}[j + 1], j \in [0, i)$，等价于，将 $\textit{nums}[j] - j, j \in [0, i)$ 操作成相同的数字。因此，先对数组 $\textit{nums}$ 作预处理，即令 $\textit{nums}[i] = \textit{nums}[i] - i$。

将区间 $[0, i]$ 范围内的数字操作成相同的数字的最小操作数，等于将区间 $[0, i]$ 范围内的数字操作成它们的中位数所需要的操作数。证明过程可以参考题目 [462. 最小操作次数使数组元素相等 II](https://leetcode.cn/problems/minimum-moves-to-equal-array-elements-ii/solutions/1501230/zui-shao-yi-dong-ci-shu-shi-shu-zu-yuan-xt3r2/)。

我们分别使用两个优先队列 $\textit{lower}$ 和 $\textit{upper}$ 来保存 $[0, i]$ 内的数字，同时使用 $\textit{lowerSum}$ 和 $\textit{upperSum}$ 分别保存两个优先队列的元素和，这两个优先队列中的元素满足以下两个条件：

1. 优先队列 $\textit{lower}$ 保存的任一元素都小于等于优先队列 $\textit{upper}$ 保存的任一元素；

2. 优先队列 $\textit{lower}$ 的元素数目 $n_\textit{lower}$ 与优先队列 $\textit{upper}$ 的元素数目 $n_\textit{upper}$ 满足 $n_\textit{upper} \le n_\textit{lower} \le n_\textit{upper} + 1$。

遍历数组 $\textit{nums}$，假设当前遍历到元素 $\textit{nums}[i]$，考虑如何将元素 $\textit{nums}[i]$ 加入优先队列，同时不违反以上条件。首先如果 $\textit{lower}$ 为空或 $\textit{nums}[i]$ 小于 $\textit{lower}$ 的最大元素，那么我们将 $\textit{nums}[i]$ 加入 $\textit{lower}$，更新 $\textit{lowerSum}$，否则将 $\textit{nums}[i]$ 加入 $\textit{upper}$ 中，更新 $\textit{upperSum}$，此时条件 $1$ 依旧满足。然后我们需要调整优先队列的元素数目关系，以满足条件 $2$：

- 如果 $n_\textit{lower} \gt n_\textit{upper}$，那么将 $\textit{lower}$ 的最大值移动到 $\textit{upper}$，同时更新 $\textit{lowerSum}$ 和 $\textit{upperSum}$。

- 如果 $n_\textit{lower} \lt n_\textit{upper}$，那么将 $\textit{upper}$ 的最小值移动到 $\textit{lower}$，同时更新 $\textit{lowerSum}$ 和 $\textit{upperSum}$。

那么：

- 当 $i+1$ 为偶数时，令中位数为 $t$，那么有 $\max(\textit{lower}) \le t \le \min(\textit{upper})$，从而 $\textit{res}_i = \sum_{j=0}^{i}|\textit{nums}[j] - t| = \textit{upperSum} - \textit{lowerSum}$。

- 当 $i+1$ 为奇数时，中位数 $t = \max(\textit{lower})$，$\textit{res}_i = \sum_{j=0}^{i}|\textit{nums}[j] - t| = \textit{upperSum} - \textit{lowerSum} + \max(\textit{lower})$。

返回结果数组 $\textit{res}$ 即可。

> 类似的求解中位数的题目有 [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/description/)

```C++ [sol1-C++]
class Solution {
public:
    vector<int> numsGame(vector<int>& nums) {
        int n = nums.size();
        vector<int> res(n);
        priority_queue<int> lower;
        priority_queue<int, vector<int>, greater<>> upper;
        long long mod = 1e9 + 7;
        long long lowerSum = 0, upperSum = 0;
        for (int i = 0; i < n; i++) {
            int x = nums[i] - i;
            if (lower.empty() || lower.top() >= x) {
                lowerSum += x;
                lower.push(x);
                if (lower.size() > upper.size() + 1) {
                    upperSum += lower.top();
                    upper.push(lower.top());
                    lowerSum -= lower.top();
                    lower.pop();
                }
            } else {
                upperSum += x;
                upper.push(x);
                if (lower.size() < upper.size()) {
                    lowerSum += upper.top();
                    lower.push(upper.top());
                    upperSum -= upper.top();
                    upper.pop();
                }
            }
            if ((i + 1) % 2 == 0) {
                res[i] = (upperSum - lowerSum) % mod;
            } else {
                res[i] = (upperSum - lowerSum + lower.top()) % mod;
            }
        }
        return res;
    }
};
```

```Java [sol1-Java]
class Solution {
    public int[] numsGame(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        PriorityQueue<Integer> lower = new PriorityQueue<Integer>((a, b) -> b - a);
        PriorityQueue<Integer> upper = new PriorityQueue<Integer>((a, b) -> a - b);
        final int MOD = 1000000007;
        long lowerSum = 0, upperSum = 0;
        for (int i = 0; i < n; i++) {
            int x = nums[i] - i;
            if (lower.isEmpty() || lower.peek() >= x) {
                lowerSum += x;
                lower.offer(x);
                if (lower.size() > upper.size() + 1) {
                    upperSum += lower.peek();
                    upper.offer(lower.peek());
                    lowerSum -= lower.peek();
                    lower.poll();
                }
            } else {
                upperSum += x;
                upper.offer(x);
                if (lower.size() < upper.size()) {
                    lowerSum += upper.peek();
                    lower.offer(upper.peek());
                    upperSum -= upper.peek();
                    upper.poll();
                }
            }
            if ((i + 1) % 2 == 0) {
                res[i] = (int) ((upperSum - lowerSum) % MOD);
            } else {
                res[i] = (int) ((upperSum - lowerSum + lower.peek()) % MOD);
            }
        }
        return res;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int[] NumsGame(int[] nums) {
        int n = nums.Length;
        int[] res = new int[n];
        PriorityQueue<int, int> lower = new PriorityQueue<int, int>();
        PriorityQueue<int, int> upper = new PriorityQueue<int, int>();
        const int MOD = 1000000007;
        long lowerSum = 0, upperSum = 0;
        for (int i = 0; i < n; i++) {
            int x = nums[i] - i;
            if (lower.Count == 0 || lower.Peek() >= x) {
                lowerSum += x;
                lower.Enqueue(x, -x);
                if (lower.Count > upper.Count + 1) {
                    upperSum += lower.Peek();
                    upper.Enqueue(lower.Peek(), lower.Peek());
                    lowerSum -= lower.Peek();
                    lower.Dequeue();
                }
            } else {
                upperSum += x;
                upper.Enqueue(x, x);
                if (lower.Count < upper.Count) {
                    lowerSum += upper.Peek();
                    lower.Enqueue(upper.Peek(), -upper.Peek());
                    upperSum -= upper.Peek();
                    upper.Dequeue();
                }
            }
            if ((i + 1) % 2 == 0) {
                res[i] = (int) ((upperSum - lowerSum) % MOD);
            } else {
                res[i] = (int) ((upperSum - lowerSum + lower.Peek()) % MOD);
            }
        }
        return res;
    }
}
```

```Go [sol1-Go]
type BasePQ struct {
    sort.IntSlice
}

func (pq *BasePQ) Push(x any) {
    pq.IntSlice = append(pq.IntSlice, x.(int))
}

func (pq *BasePQ) Pop() any {
    n := len(pq.IntSlice)
    x := pq.IntSlice[n - 1]
    pq.IntSlice = pq.IntSlice[:n-1]
    return x
}

func (pq *BasePQ) Top() int {
    return pq.IntSlice[0]
}

type MinPQ struct {
    *BasePQ
}

func (pq *MinPQ) Less(i, j int) bool {
    return pq.BasePQ.Less(i, j)
}

type MaxPQ struct {
    *BasePQ
}

func (pq *MaxPQ) Less(i, j int) bool {
    return pq.BasePQ.Less(j, i)
}

func numsGame(nums []int) []int {
    n := len(nums)
    res := make([]int, n)
    lower, upper := &MaxPQ{&BasePQ{}}, &MinPQ{&BasePQ{}}
    lowerSum, upperSum := int64(0), int64(0)
    mod := int64(1e9 + 7)
    for i := 0; i < n; i++ {
        x := nums[i] - i
        if lower.Len() == 0 || lower.Top() >= x {
            lowerSum += int64(x)
            heap.Push(lower, x)
            if lower.Len() > upper.Len() + 1 {
                upperSum += int64(lower.Top())
                heap.Push(upper, lower.Top())
                lowerSum -= int64(heap.Pop(lower).(int))
            }
        } else {
            upperSum += int64(x)
            heap.Push(upper, x)
            if lower.Len() < upper.Len() {
                lowerSum += int64(upper.Top())
                heap.Push(lower, upper.Top())
                upperSum -= int64(heap.Pop(upper).(int))
            }
        }
        if (i + 1) % 2 == 0 {
            res[i] = int((upperSum - lowerSum) % mod)
        } else {
            res[i] = int((upperSum - lowerSum + int64(lower.Top())) % mod)
        }
    }
    return res
}
```

```Python [sol1-Python3]
class Solution:
    def numsGame(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        lower, upper = [], []
        lowerSum, upperSum = 0, 0
        mod = int(1e9 + 7)
        for i in range(n):
            x = nums[i] - i
            if len(lower) == 0 or -lower[0] >= x:
                lowerSum += x
                heappush(lower, -x)
                if len(lower) > len(upper) + 1:
                    upperSum -= lower[0]
                    heappush(upper, -lower[0])
                    lowerSum += heappop(lower)
            else:
                upperSum += x
                heappush(upper, x)
                if len(lower) < len(upper):
                    lowerSum += upper[0]
                    heappush(lower, -upper[0])
                    upperSum -= heappop(upper)
            if (i + 1) % 2 == 0:
                res[i] = (upperSum - lowerSum) % mod
            else:
                res[i] = (upperSum - lowerSum - lower[0]) % mod
        return res
```

```C [sol1-C]
typedef bool (*LessFunc)(int, int);

typedef struct {
    int *data;
    int size;
    LessFunc cmp;
} Heap;

void heapInit(Heap *this, int n, LessFunc cmp) {
    this->data = (int *)malloc(sizeof(int) * n);
    this->size = 0;
    this->cmp = cmp;
}

void heapFree(Heap *this) {
    free(this->data);
}

void swap(Heap *this, int i, int j) {
    int x = this->data[i];
    this->data[i] = this->data[j];
    this->data[j] = x;
}

void down(Heap *this, int i) {
    for (int k = 2 * i + 1; k < this->size; k = 2 * k + 1) {
        // 父节点 (k - 1) / 2，左子节点 k，右子节点 k + 1
        if (k + 1 < this->size && this->cmp(this->data[k], this->data[k + 1])) {
            k++;
        }
        if (!this->cmp(this->data[(k - 1) / 2], this->data[k])) {
            break;
        }
        swap(this, k, (k - 1) / 2);
    }
}

void push(Heap *this, int x) {
    this->data[this->size] = x;
    this->size++;
    for (int i = this->size - 1; i > 0 && this->cmp(this->data[(i - 1) / 2], this->data[i]); i = (i - 1) / 2) {
        swap(this, i, (i - 1) / 2);
    }
}

int pop(Heap *this) {
    swap(this, 0, this->size - 1);
    this->size--;
    down(this, 0);
    return this->data[this->size];
}

int size(Heap *this) {
    return this->size;
}

int top(Heap *this) {
    return this->data[0];
}

bool less(int x, int y) {
    return x < y;
}

bool greater(int x, int y) {
    return x > y;
}

int* numsGame(int *nums, int numsSize, int *returnSize) {
    int n = numsSize;
    int *res = (int *)malloc(sizeof(int) * n);
    Heap lower, upper;
    heapInit(&lower, n, less);
    heapInit(&upper, n, greater);
    long long mod = 1e9 + 7;
    long long lowerSum = 0, upperSum = 0;
    for (int i = 0; i < n; i++) {
        int x = nums[i] - i;
        if (size(&lower) == 0 || top(&lower) >= x) {
            lowerSum += x;
            push(&lower, x);
            if (size(&lower) > size(&upper) + 1) {
                upperSum += top(&lower);
                push(&upper, top(&lower));
                lowerSum -= pop(&lower);
            }
        } else {
            upperSum += x;
            push(&upper, x);
            if (size(&lower) < size(&upper)) {
                lowerSum += top(&upper);
                push(&lower, top(&upper));
                upperSum -= pop(&upper);
            }
        }
        if ((i + 1) % 2 == 0) {
            res[i] = (upperSum - lowerSum) % mod;
        } else {
            res[i] = (upperSum - lowerSum + top(&lower)) % mod;
        }
    }

    heapFree(&lower);
    heapFree(&upper);
    *returnSize = n;
    return res;
}
```

```JavaScript [sol1-JavaScript]
var numsGame = function(nums) {
    const n = nums.length;
    const res = new Array(n).fill(0);
    const lower = new MaxPriorityQueue();
    const upper = new MinPriorityQueue();
    const mod = 1e9 + 7;
    let lowerSum = 0, upperSum = 0;
    for (let i = 0; i < n; i++) {
        const x = nums[i] - i;
        if (lower.size() == 0 || lower.front().element >= x) {
            lowerSum += x;
            lower.enqueue(x);
            if (lower.size() > upper.size() + 1) {
                upperSum += lower.front().element;
                upper.enqueue(lower.front().element);
                lowerSum -= lower.front().element;
                lower.dequeue();
            }
        } else {
            upperSum += x;
            upper.enqueue(x);
            if (lower.size() < upper.size()) {
                lowerSum += upper.front().element;
                lower.enqueue(upper.front().element);
                upperSum -= upper.front().element;
                upper.dequeue();
            }
        }
        if ((i + 1) % 2 == 0) {
            res[i] = (upperSum - lowerSum) % mod;
        } else {
            res[i] = (upperSum - lowerSum + lower.front().element) % mod;
        }
    }
    return res;
};
```

```TypeScript [sol1-TypeScript]
function numsGame(nums: number[]): number[] {
    const n: number = nums.length;
    const res = new Array(n).fill(0);
    const lower = new MaxPriorityQueue();
    const upper = new MinPriorityQueue();
    const mod: number = 1e9 + 7;
    let lowerSum = 0, upperSum = 0;
    for (let i = 0; i < n; i++) {
        const x = nums[i] - i;
        if (lower.size() == 0 || lower.front().element >= x) {
            lowerSum += x;
            lower.enqueue(x);
            if (lower.size() > upper.size() + 1) {
                upperSum += lower.front().element;
                upper.enqueue(lower.front().element);
                lowerSum -= lower.front().element;
                lower.dequeue();
            }
        } else {
            upperSum += x;
            upper.enqueue(x);
            if (lower.size() < upper.size()) {
                lowerSum += upper.front().element;
                lower.enqueue(upper.front().element);
                upperSum -= upper.front().element;
                upper.dequeue();
            }
        }
        if ((i + 1) % 2 == 0) {
            res[i] = (upperSum - lowerSum) % mod;
        } else {
            res[i] = (upperSum - lowerSum + lower.front().element) % mod;
        }
    }
    return res;
};
```

**复杂度分析**

- 时间复杂度：$O(n \log n)$, 其中 $n$ 是数组 $nums$ 的长度。优先队列入队、出队都需要 $O(\log n)$ 的时间，总共需要 $O(n \log n)$ 的时间。

- 空间复杂度：$O(n)$。