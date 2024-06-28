#### 方法一：动态规划

**思路与算法**

我们可以使用动态规划解决本题。

对于第 $i$ 位油漆匠，我们可以让他付费或者免费工作：

- 如果付费工作，我们花费 $\textit{cost}[i]$ 的钱，但可以得到 $\textit{time}[i]$ 次让其余油漆匠免费工作的机会；

- 如果免费工作，我们付出的代价为 $1$ 次免费工作的次数。

我们的目标是最小化付出的钱，而变量是每一位油漆匠是付费还是免费工作，以及可以免费工作的次数，因此可以设计状态为：$f(i, j)$ 表示我们考虑了前 $i$ 位（第 $0 \sim i-1$ 位）油漆匠，并且免费工作的次数为 $j$ 时的最少开销。状态转移方程只需要考虑第 $i$ 位油漆匠是付费还是免费工作：

- 如果付费工作，那么有：
    $$
    f(i + 1, j + \textit{time}[i]) \leftarrow f(i, j) + \textit{cost}[i]
    $$

- 如果免费工作，那么有：
    $$
    f(i + 1, j - 1) \leftarrow f(i, j)
    $$

在进行状态转移时，我们将 $\leftarrow$ 左侧的状态更新为其与右侧状态的最小值。初始时，所有状态的值均为 $+\infty$，只有 $f(0, 0) = 0$。最终的答案即为所有满足 $j \geq 0$ 的 $f(n, j)$ 中的最小值。

**细节**

对于上述的状态转移方程，有一个重要的问题是：$j$ 的枚举范围是什么？可能一开始会想到的范围是 $[0, \sum_i \textit{time}[i]]$，但这是不合理的：

- 时间复杂度较高，为 $O(n \times (\sum_i \textit{time}[i]))$，需要进行优化；

- 当 $j = 0$ 时，在状态转移方程中我们无法油漆匠免费工作，但实际上是可以的，只需要后续的油漆匠付费工作即可，如果让 $j$ 的范围仅为非负数，我们并不能考虑到左右的情况。

因此，我们需要引入 $j < 0$ 的范围：免费工作的次数是可以「赊账」的，只需要后续油漆匠来付费工作补全这个次数即可。因此一个合理的枚举 $j$ 的范围为：

- $j$ 的下限为 $-n$。在极端情况下，所有油漆匠都免费工作；

- $j$ 的上限为 $n$。这是因为付费工作的目的是为了让其他的油漆匠可以免费工作，而一共只有 $n$ 位油漆匠，所以免费工作的机会超过 $n$ 是没有意义的。

因此，状态转移方程变为：

$$
f(i + 1, \min(j + \textit{time}[i], n)) \leftarrow f(i, j) + \textit{cost}[i]
$$

以及：

$$
f(i + 1, j - 1) \leftarrow f(i, j) \quad (j \neq -n)
$$

在代码实现时，我们可以使用二维数组来存储 $f(i, j)$。但对于大部分语言，是不支持负数下标的，因此我们需要给 $j$ 一个 $+n$ 的偏移，将它的范围从 $[-n, n]$ 变为 $[0, 2n]$。

**优化**

注意到 $f(i, j)$ 只会转移到 $f(i + 1, \cdots)$，因此可以使用两个一维数组代替二维数组进行状态转移，减少使用的空间。

**代码**

```C++ [sol1-C++]
class Solution {
public:
    int paintWalls(vector<int>& cost, vector<int>& time) {
        int n = cost.size();
        vector<int> f(n * 2 + 1, INT_MAX / 2);
        f[n] = 0;
        for (int i = 0; i < n; ++i) {
            vector<int> g(n * 2 + 1, INT_MAX / 2);
            for (int j = 0; j <= n * 2; ++j) {
                // 付费
                g[min(j + time[i], n * 2)] = min(g[min(j + time[i], n * 2)], f[j] + cost[i]);
                // 免费
                if (j > 0) {
                    g[j - 1] = min(g[j - 1], f[j]);
                }
            }
            f = move(g);
        }
        return *min_element(f.begin() + n, f.end());
    }
};
```

```Java [sol1-Java]
class Solution {
    public int paintWalls(int[] cost, int[] time) {
        int n = cost.length;
        int[] f = new int[n * 2 + 1];
        Arrays.fill(f, Integer.MAX_VALUE / 2);
        f[n] = 0;
        for (int i = 0; i < n; ++i) {
            int[] g = new int[n * 2 + 1];
            Arrays.fill(g, Integer.MAX_VALUE / 2);
            for (int j = 0; j <= n * 2; ++j) {
                // 付费
                g[Math.min(j + time[i], n * 2)] = Math.min(g[Math.min(j + time[i], n * 2)], f[j] + cost[i]);
                // 免费
                if (j > 0) {
                    g[j - 1] = Math.min(g[j - 1], f[j]);
                }
            }
            f = g;
        }
        int ans = f[n];
        for (int i = n + 1; i <= n * 2; i++) {
            ans = Math.min(ans, f[i]);
        }
        return ans;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int PaintWalls(int[] cost, int[] time) {
        int n = cost.Length;
        int[] f = new int[n * 2 + 1];
        Array.Fill(f, int.MaxValue / 2);
        f[n] = 0;
        for (int i = 0; i < n; ++i) {
            int[] g = new int[n * 2 + 1];
            Array.Fill(g, int.MaxValue / 2);
            for (int j = 0; j <= n * 2; ++j) {
                // 付费
                g[Math.Min(j + time[i], n * 2)] = Math.Min(g[Math.Min(j + time[i], n * 2)], f[j] + cost[i]);
                // 免费
                if (j > 0) {
                    g[j - 1] = Math.Min(g[j - 1], f[j]);
                }
            }
            f = g;
        }
        int ans = f[n];
        for (int i = n + 1; i <= n * 2; i++) {
            ans = Math.Min(ans, f[i]);
        }
        return ans;
    }
}
```

```Python [sol1-Python3]
class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        f = [inf] * (n * 2 + 1)
        f[n] = 0
        for (cost_i, time_i) in zip(cost, time):
            g = [inf] * (n * 2 + 1)
            for j in range(n * 2 + 1):
                # 付费
                g[min(j + time_i, n * 2)] = min(g[min(j + time_i, n * 2)], f[j] + cost_i)
                # 免费
                if j > 0:
                    g[j - 1] = min(g[j - 1], f[j])
            f = g
        return min(f[n:])
```

```Go [sol1-Go]
func paintWalls(cost []int, time []int) int {
    n := len(cost)
    f := make([]int, 2 * n + 1)
    for k := range f {
        f[k] = math.MaxInt / 2
    }
    f[n] = 0
    for i := 0; i < n; i++ {
        g := make([]int, n * 2 + 1)
        for k := range g {
            g[k] = math.MaxInt / 2
        }
        for j := 0; j <= n * 2; j++ {
            // 付费
            g[min(j + time[i], n * 2)] = min(g[min(j + time[i], n * 2)], f[j] + cost[i])
            // 免费
            if j > 0 {
                g[j - 1] = min(g[j - 1], f[j])
            }
        }
        f = g
    }
    res := math.MaxInt
    for i := n; i < len(f); i++ {
        res = min(res, f[i])
    }
    return res
}
```

```C [sol1-C]
int min(int a, int b) {
    return a < b ? a : b;
}

int paintWalls(int *cost, int costSize, int *time, int timeSize){
    int n = costSize;
    int *f = (int *)malloc((n * 2 + 1) * sizeof(int));
    for (int i = 0; i <= 2 * n; i++) {
        f[i] = INT_MAX / 2;
    }
    f[n] = 0;
    for (int i = 0; i < n; ++i) {
        int *g = (int *)malloc((n * 2 + 1) * sizeof(int));
        for (int i = 0; i <= 2 * n; i++) {
            g[i] = INT_MAX / 2;
        }
        for (int j = 0; j <= n * 2; ++j) {
            // 付费
            g[min(j + time[i], n * 2)] = min(g[min(j + time[i], n * 2)], f[j] + cost[i]);
            // 免费
            if (j > 0) {
                g[j - 1] = min(g[j - 1], f[j]);
            }
        }
        free(f);
        f = g;
    }
    int res = INT_MAX;
    for (int i = n; i <= 2 * n; i++) {
        res = min(res, f[i]);
    }
    free(f);
    return res;
}
```

```JavaScript [sol1-JavaScript]
var paintWalls = function(cost, time) {
    let n = cost.length;
    let f = new Array(n * 2 + 1).fill(Number.MAX_SAFE_INTEGER / 2);
    f[n] = 0;
    for (let i = 0; i < n; ++i) {
        let g = new Array(n * 2 + 1).fill(Number.MAX_SAFE_INTEGER / 2);
        for (let j = 0; j <= n * 2; ++j) {
            // 付费
            g[Math.min(j + time[i], n * 2)] = Math.min(g[Math.min(j + time[i], n * 2)], f[j] + cost[i]);
            // 免费
            if (j > 0) {
                g[j - 1] = Math.min(g[j - 1], f[j]);
            }
        }
        f = g.slice();
    }
    return Math.min(...f.slice(n));
};
```

```TypeScript [sol1-TypeScript]
function paintWalls(cost: number[], time: number[]): number {
    let n: number = cost.length;
    let f: number[] = new Array(n * 2 + 1).fill(Number.MAX_SAFE_INTEGER / 2);
    f[n] = 0;
    for (let i = 0; i < n; ++i) {
        let g: number[] = new Array(n * 2 + 1).fill(Number.MAX_SAFE_INTEGER / 2);
        for (let j = 0; j <= n * 2; ++j) {
            // 付费
            g[Math.min(j + time[i], n * 2)] = Math.min(g[Math.min(j + time[i], n * 2)], f[j] + cost[i]);
            // 免费
            if (j > 0) {
                g[j - 1] = Math.min(g[j - 1], f[j]);
            }
        }
        f = [...g];
    }
    return Math.min(...f.slice(n));
};
```

```Rust [sol1-Rust]
use std::cmp::min;

impl Solution {
    pub fn paint_walls(cost: Vec<i32>, time: Vec<i32>) -> i32 {
        let n = cost.len();
        let mut f = vec![i32::MAX / 2; n * 2 + 1];
        f[n] = 0;
        for i in 0..n {
            let mut g = vec![i32::MAX / 2; n * 2 + 1];
            for j in 0..=n * 2 {
                // 付费
                g[min(j + time[i] as usize, n * 2)] = min(g[min(j + time[i] as usize, n * 2)], f[j] + cost[i]);
                // 免费
                if j > 0 {
                    g[j - 1] = min(g[j - 1], f[j]);
                }
            }
            f = g;
        }
        *f.iter().skip(n).min().unwrap()
    }
}
```


**复杂度分析**

- 时间复杂度：$O(n^2)$。

- 空间复杂度：$O(n)$。

#### 方法二：额外的空间优化

**思路与算法**

记 $g(i, j) = f(i + 1, j - 1)$，那么方法一中的状态转移方程变为：

$$
g(i, \min(j + \textit{time}[i], n) + 1) \leftarrow f(i, j) + \textit{cost}[i]
$$

以及：

$$
g(i, j) \leftarrow f(i, j) \quad (j \neq -n)
$$

如果还是使用方法一中两个一维数组进行转移，那么免费工作的情况相当于把数组拷贝了一份，而付费工作的情况是一个普通的状态转移方程，并且 $g$ 中的 $j$ 维度总是大于等于 $f$ 中的 $j$ 维度。

因此这是一个类似背包问题的状态转移方程，我们只需要使用一个一维数组进行状态转移即可。对于 $f(i, j)$，$j$ 更准确的范围是 $[-i, i]$，而每一次 $g(i, j) = f(i + 1, j - 1)$ 会使得 $j$ 有 $+1$ 的偏移，与 $[-i, i]$ 中的下限 $-i$ 正好抵消，因此新的状态转移方程中，$j$ 的下限为 $0$，上限可以通过付费工作的状态转移方程看出是 $n + 1$。因此状态转移方程仅为：

$$
f(\min(j + \textit{time}[i], n) + 1) \leftarrow f(j) + \textit{cost}[i]
$$

最终的答案为 $f(n)$ 和 $f(n + 1)$ 中的较小值，这是因为方法一中的答案是 $f(n, j) ~ (j \geq 0)$ 中的最小值，而进行了 $n$ 次 $j$ 的 $+1$ 偏移后，就变为方法二中 $f(j) ~(j \geq n)$ 的最小值，由于 $j$ 的上限为 $n + 1$，因此只有 $f(n)$ 和 $f(n + 1)$ 才能作为答案。

**代码**

```C++ [sol2-C++]
class Solution {
public:
    int paintWalls(vector<int>& cost, vector<int>& time) {
        int n = cost.size();
        vector<int> f(n + 2, INT_MAX / 2);
        f[0] = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = n + 1; j >= 0; --j) {
                f[min(j + time[i], n) + 1] = min(f[min(j + time[i], n) + 1], f[j] + cost[i]);
            }
        }
        return min(f[n], f[n + 1]);
    }
};
```

```Java [sol2-Java]
class Solution {
    public int paintWalls(int[] cost, int[] time) {
        int n = cost.length;
        int[] f = new int[n + 2];
        Arrays.fill(f, Integer.MAX_VALUE / 2);
        f[0] = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = n + 1; j >= 0; --j) {
                f[Math.min(j + time[i], n) + 1] = Math.min(f[Math.min(j + time[i], n) + 1], f[j] + cost[i]);
            }
        }
        return Math.min(f[n], f[n + 1]);
    }
}
```

```C# [sol2-C#]
public class Solution {
    public int PaintWalls(int[] cost, int[] time) {
        int n = cost.Length;
        int[] f = new int[n + 2];
        Array.Fill(f, int.MaxValue / 2);
        f[0] = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = n + 1; j >= 0; --j) {
                f[Math.Min(j + time[i], n) + 1] = Math.Min(f[Math.Min(j + time[i], n) + 1], f[j] + cost[i]);
            }
        }
        return Math.Min(f[n], f[n + 1]);
    }
}
```

```Python [sol2-Python3]
class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        f = [0] + [inf] * (n + 1)
        for (cost_i, time_i) in zip(cost, time):
            for j in range(n + 1, -1, -1):
                f[min(j + time_i, n) + 1] = min(f[min(j + time_i, n) + 1], f[j] + cost_i)
        return min(f[n], f[n + 1])
```

```Go [sol2-Go]
func paintWalls(cost []int, time []int) int {
    n := len(cost)
    f := make([]int, n + 2)
    for k := range f {
        f[k] = math.MaxInt / 2
    }
    f[0] = 0
    for i := 0; i < n; i++ {
        for j := n + 1; j >= 0; j-- {
            f[min(j + time[i], n) + 1] = min(f[min(j + time[i], n) + 1], f[j] + cost[i])
        }
    }
    return min(f[n], f[n + 1])
}
```

```C [sol2-C]
int min(int a, int b) {
    return a < b ? a : b;
}

int paintWalls(int *cost, int costSize, int *time, int timeSize){
    int n = costSize;
    int *f = (int *)malloc((n + 2) * sizeof(int));
    for (int i = 0; i < n + 2; i++) {
        f[i] = INT_MAX / 2;
    }
    f[0] = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = n + 1; j >= 0; --j) {
            f[min(j + time[i], n) + 1] = min(f[min(j + time[i], n) + 1], f[j] + cost[i]);
        }
    }
    return min(f[n], f[n + 1]);
}
```

```JavaScript [sol2-JavaScript]
var paintWalls = function(cost, time) {
     let n = cost.length;
    let f = new Array(n + 2).fill(Number.MAX_SAFE_INTEGER / 2);
    f[0] = 0;
    for (let i = 0; i < n; ++i) {
        for (let j = n + 1; j >= 0; --j) {
            f[Math.min(j + time[i], n) + 1] = Math.min(f[Math.min(j + time[i], n) + 1], f[j] + cost[i]);
        }
    }
    return Math.min(f[n], f[n + 1]);
};
```

```TypeScript [sol2-TypeScript]
function paintWalls(cost: number[], time: number[]): number {
    let n: number = cost.length;
    let f: number[] = new Array(n + 2).fill(Number.MAX_SAFE_INTEGER / 2);
    f[0] = 0;
    for (let i = 0; i < n; ++i) {
        for (let j = n + 1; j >= 0; --j) {
            f[Math.min(j + time[i], n) + 1] = Math.min(f[Math.min(j + time[i], n) + 1], f[j] + cost[i]);
        }
    }
    return Math.min(f[n], f[n + 1]);
};
```

```Rust [sol2-Rust]
use std::cmp::min;

impl Solution {
    pub fn paint_walls(cost: Vec<i32>, time: Vec<i32>) -> i32 {
        let n = cost.len();
        let mut f = vec![i32::MAX / 2; n + 2];
        f[0] = 0;
        for i in 0..n {
            for j in (0..=n).rev() {
                f[min(j + time[i] as usize, n) + 1] = min(f[min(j + time[i] as usize, n) + 1], f[j] + cost[i]);
            }
        }
        min(f[n], f[n + 1])
    }
}
```

**复杂度分析**

- 时间复杂度：$O(n^2)$。

- 空间复杂度：$O(n)$，但相对于方法一只需要一个一维数组。