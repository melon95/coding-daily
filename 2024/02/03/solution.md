#### 方法一：记忆化搜索

**思路与算法**

根据题意可知有 $n$ 块石子，游戏的每个回合中，选手可以从行中**移除**最左边的石头或最右边的石头，并获得与该行中剩余石头值之**和**相等的得分。由于先手的 $\text{Bob}$ 一定会输，此时 $\text{Bob}$ 与 $\text{Alice}$ 得分的差值一定小于 $0$；$\text{Alice}$ 一定会赢，此时 $\text{Alice}$ 与 $\text{Bob}$ 得分的差值一定大于 $0$，$\text{Bob}$ 尽力**减小**得分的差值，$\text{Alice}$ 的尽量**扩大**得分的差值，二者的博弈过程即等价于不管是 $\text{Bob}$ 还是 $\text{Alice}$ 都尽可能的**扩大得分**的差值。根据题意可以推出，假设当前石头序列固定，则不管是 $\text{Alice}$ 先手还是 $\text{Bob}$ 先手，二者之间得分的**最大差值**一定是确定的。

假设当前只剩下索引区间 $[i,j]$ 的石头待选择，此时对于该轮选手来说，它的最优解到底是该选最左侧的 $i$ 还是右侧的元素 $j$ 呢？假设当前只剩下 $[s_i,s_{i+1},s_{i+2},s_{i+3},\cdots,s_j]$，且此时轮到 $\text{Bob}$ 选择，$\text{Bob}$ 有两种选择：

+ 假设 $\text{Bob}$ 首先拿走的是 $s_i$，则其得分为 $\sum_{k=i+1}^{j}$，剩余的元素为 $[s_{i+1},s_{i+2},s_{i+3},\cdots,s_j]$，由于 $\text{Alice}$ 想努力的扩大得分差值，假设 $\text{Alice}$ 在剩余石头序列 $[s_{i+1},s_{i+2},s_{i+3},\cdots,s_j]$ 中游戏的得分与 $\text{Bob}$ 得分最大差值为 $f(i+1,j)$，此时 $\text{Bob}$ 与 $\text{Alice}$ 的得分差值即为 $\sum_{k=i+1}^{j} - f(i+1,j)$；

+ 假设 $\text{Bob}$ 首先拿走的是 $s_j$，则其得分为 $\sum_{k=i}^{j-1}$，剩余的元素为 $[s_{i},s_{i+1},s_{i+2},\cdots,s_{j-1}]$，由于 $\text{Alice}$ 想努力的扩大得分差值，假设 $\text{Alice}$ 在剩余石头序列 $[s_{i},s_{i+1},s_{i+2},\cdots,s_{j-1}]$ 中游戏的得分与 $\text{Bob}$ 得分最大差值为 $f(i,j-1)$，此时 $\text{Bob}$ 与 $\text{Alice}$ 的得分差值即为 $\sum_{k=i}^{j-1} - f(i,j-1)$；

+ 此时可以知道序列 $[s_i,s_{i+1},s_{i+2},s_{i+3},\cdots,s_j]$ 中，$\text{Bob}$ 与 $\text{Alice}$ **得分差的最大值** 为 $f(i,j) = \max(\sum_{k=i+1}^{j} - f(i+1,j), \sum_{k=i}^{j-1} - f(i,j-1))$，综上可以得知**最大得分差**可以理解为此次操作之后，当前选手所收获的价值 - 剩余序列中对手与自己的得分差的最大值。

根据上述推论，我们可以采用自顶向下的记忆化搜索，如要求得区间 $[i,j]$ 得分的最大差值为 $f(i,j)$，此时根据上述的分析需要求出子区间 $[i,j-1]$ 与 $[i+1,j]$ 的最优值即可。由于每次移除石头的得分为区间的累加和，此时我们可以维护序列的前缀和，即可快速求出区间和。

**代码**

```C++ [sol1-C++]
class Solution {
public:
    int stoneGameVII(vector<int>& stones) {
        int n = stones.size();
        vector<int> sum(n + 1);
        vector<vector<int>> memo(n, vector<int>(n, 0));
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + stones[i];
        }

        function<int(int, int)> dfs = [&](int i, int j) -> int {
            if (i >= j) {
                return 0;
            }
            if (memo[i][j] != 0) {
                return memo[i][j];
            }
            int res = max(sum[j + 1] - sum[i + 1] - dfs(i + 1, j), sum[j] - sum[i] - dfs(i, j - 1));
            memo[i][j] = res;
            return res;
        };
        return dfs(0, n - 1);
    }
};
```

```Java [sol1-Java]
class Solution {
    public int stoneGameVII(int[] stones) {
        int n = stones.length;
        int[] sum = new int[n + 1];
        int[][] memo = new int[n][n];
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + stones[i];
        }
        return dfs(0, n - 1, sum, memo);
    }

    public int dfs(int i, int j, int[] sum, int[][] memo) {
        if (i >= j) {
            return 0;
        }
        if (memo[i][j] != 0) {
            return memo[i][j];
        }
        int res = Math.max(sum[j + 1] - sum[i + 1] - dfs(i + 1, j, sum, memo), sum[j] - sum[i] - dfs(i, j - 1, sum, memo));
        memo[i][j] = res;
        return res;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int StoneGameVII(int[] stones) {
        int n = stones.Length;
        int[] sum = new int[n + 1];
        int[][] memo = new int[n][];
        for (int i = 0; i < n; i++) {
            memo[i] = new int[n];
        }
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + stones[i];
        }
        return DFS(0, n - 1, sum, memo);
    }

    public int DFS(int i, int j, int[] sum, int[][] memo) {
        if (i >= j) {
            return 0;
        }
        if (memo[i][j] != 0) {
            return memo[i][j];
        }
        int res = Math.Max(sum[j + 1] - sum[i + 1] - DFS(i + 1, j, sum, memo), sum[j] - sum[i] - DFS(i, j - 1, sum, memo));
        memo[i][j] = res;
        return res;
    }
}
```

```C [sol1-C]
int dfs(int i, int j, int **memo, int *sum) {
    if (i >= j) {
        return 0;
    }
    if (memo[i][j] != 0) {
        return memo[i][j];
    }
    int res = fmax(sum[j + 1] - sum[i + 1] - dfs(i + 1, j, memo, sum), \
                   sum[j] - sum[i] - dfs(i, j - 1, memo, sum));
    memo[i][j] = res;
    return res;
}

int stoneGameVII(int* stones, int stonesSize) {
    int n = stonesSize;
    int *sum = (int *)malloc(sizeof(int) * (n + 1));
    int **memo = (int *)malloc(sizeof(int *) * n);
    memset(sum, 0, sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + stones[i];
        memo[i] = (int *)malloc(sizeof(int) * n);
        memset(memo[i], 0, sizeof(int) * n);
    }
    int res = dfs(0, n - 1, memo, sum);
    for (int i = 0; i < n; i++) {
        free(memo[i]);
    }
    free(memo);
    free(sum);
    return res;
}
```

```Python [sol1-Python3]
class Solution:
    def stoneGameVII(self, stones: List[int]) -> int:
        pre = [0]
        for s in stones:
            pre.append(pre[-1] + s)
        
        @cache
        def dfs(i, j):
            if i > j: return 0
            return max(pre[j + 1] - pre[i + 1] - dfs(i + 1, j), pre[j] - pre[i] - dfs(i, j - 1))
        res = dfs(0, len(stones) - 1)
        dfs.cache_clear()
        return res
```

```Go [sol1-Go]
func stoneGameVII(stones []int) int {
    n := len(stones)
	sum := make([]int, n + 1)
	memo := make([][]int, n)
	for i := range memo {
		memo[i] = make([]int, n)
	}

	for i := 0; i < n; i++ {
		sum[i + 1] = sum[i] + stones[i]
	}

	var dfs func(int, int) int
	dfs = func(i, j int) int {
		if i >= j {
			return 0
		}
		if memo[i][j] != 0 {
			return memo[i][j]
		}
		res := max(sum[j + 1] - sum[i + 1] - dfs(i + 1, j), sum[j] - sum[i] - dfs(i, j - 1))
		memo[i][j] = res
		return res
	}
	return dfs(0, n - 1)
}
```

```JavaScript [sol1-JavaScript]
var stoneGameVII = function(stones) {
    const n = stones.length;
    const sum = new Array(n + 1).fill(0);
    const memo = new Array(n).fill(0).map(() => new Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + stones[i];
    }

    const dfs = (i, j) => {
        if (i >= j) {
            return 0;
        }
        if (memo[i][j] !== 0) {
            return memo[i][j];
        }
        const res = Math.max(sum[j + 1] - sum[i + 1] - dfs(i + 1, j), sum[j] - sum[i] - dfs(i, j - 1));
        memo[i][j] = res;
        return res;
    };

    return dfs(0, n - 1);
};
```

```TypeScript [sol1-TypeScript]
function stoneGameVII(stones: number[]): number {
    const n: number = stones.length;
    const sum: number[] = new Array(n + 1).fill(0);
    const memo: number[][] = new Array(n).fill(0).map(() => new Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + stones[i];
    }
    const dfs = (i: number, j: number): number => {
        if (i >= j) {
            return 0;
        }
        if (memo[i][j] !== 0) {
            return memo[i][j];
        }
        const res: number = Math.max(sum[j + 1] - sum[i + 1] - dfs(i + 1, j), sum[j] - sum[i] - dfs(i, j - 1));
        memo[i][j] = res;
        return res;
    };

    return dfs(0, n - 1);
};
```

**复杂度分析**

- 时间复杂度: $O(n^2)$，其中 $n$ 表示数组的长度。求数组的前缀和需要的时间为 $O(n)$，分别需要求出每个区间 $(i,j)$ 的最优子状态，一共有 $n^2$ 个子状态需要计算，需要的时间为 $O(n^2)$，总共需要的时间为 $O(n^2)$。

- 空间复杂度：$O(n^2)$，其中 $n$ 表示数组的长度。需要存储数组的前缀和，需要的空间为 $O(n)$，需要存储每个索引 $(i,j)$ 对应的最优状态，需要的空间为 $O(n^2)$，总共需要的空间为 $O(n^2)$。

#### 方法二：动态规划

**思路与算法**

方法的一的策略还是可以采用自底向上的动态规划，我们可以首先求出区间 $[i,i]$ 的**最大得分差值**，然后不断向外扩展求出 $[i-1,i]，[i, i+1]$ 区间的最优解，一直扩展到区间 $[0,n-1]$，此时即可求出最优解返回即可。

**代码**

```C++ [sol2-C++]
class Solution {
public:
    int stoneGameVII(vector<int>& stones) {
        int n = stones.size();
        vector<int> sum(n + 1);
        vector<vector<int>> dp(n, vector<int>(n, 0));

        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + stones[i];
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = max(sum[j + 1] - sum[i + 1] - dp[i + 1][j], sum[j] - sum[i] - dp[i][j - 1]);
            }
        }

        return dp[0][n - 1];
    }
};
```

```Java [sol2-Java]
class Solution {
    public int stoneGameVII(int[] stones) {
        int n = stones.length;
        int[] sum = new int[n + 1];
        int[][] dp = new int[n][n];

        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + stones[i];
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = Math.max(sum[j + 1] - sum[i + 1] - dp[i + 1][j], sum[j] - sum[i] - dp[i][j - 1]);
            }
        }

        return dp[0][n - 1];
    }
}
```

```C# [sol2-C#]
public class Solution {
    public int StoneGameVII(int[] stones) {
        int n = stones.Length;
        int[] sum = new int[n + 1];
        int[][] dp = new int[n][];
        for (int i = 0; i < n; i++) {
            dp[i] = new int[n];
        }

        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + stones[i];
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = Math.Max(sum[j + 1] - sum[i + 1] - dp[i + 1][j], sum[j] - sum[i] - dp[i][j - 1]);
            }
        }

        return dp[0][n - 1];
    }
}
```

```C [sol2-C]
int stoneGameVII(int* stones, int stonesSize) {
    int n = stonesSize;
    int sum[n + 1];
    int dp[n][n];
    memset(dp, 0, sizeof(dp));
    memset(sum, 0, sizeof(sum));

    for (int i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + stones[i];
    }
    for (int i = n - 2; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            dp[i][j] = fmax(sum[j + 1] - sum[i + 1] - dp[i + 1][j], sum[j] - sum[i] - dp[i][j - 1]);
        }
    }
    return dp[0][n - 1];
}
```

```Python [sol2-Python]
class Solution:
    def stoneGameVII(self, stones: List[int]) -> int:
        n = len(stones)
        sum_arr = [0] * (n + 1)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            sum_arr[i + 1] = sum_arr[i] + stones[i]
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                dp[i][j] = max(sum_arr[j + 1] - sum_arr[i + 1] - dp[i + 1][j], sum_arr[j] - sum_arr[i] - dp[i][j - 1])
        return dp[0][n - 1]
```

```Go [sol2-Go]
func stoneGameVII(stones []int) int {
    n := len(stones)
    sum := make([]int, n + 1)
    dp := make([][]int, n)

    for i := range dp {
        dp[i] = make([]int, n)
    }
    for i := 0; i < n; i++ {
        sum[i + 1] = sum[i] + stones[i]
    }
    for i := n - 2; i >= 0; i-- {
        for j := i + 1; j < n; j++ {
            dp[i][j] = max(sum[j + 1]-sum[i + 1] - dp[i + 1][j], sum[j] - sum[i] - dp[i][j - 1])
        }
    }
    return dp[0][n-1]
}
```

```JavaScript [sol2-JavaScript]
var stoneGameVII = function(stones) {
    const n = stones.length;
    const sum = new Array(n + 1).fill(0);
    const dp = new Array(n).fill(0).map(() => new Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + stones[i];
    }
    for (let i = n - 2; i >= 0; i--) {
        for (let j = i + 1; j < n; j++) {
            dp[i][j] = Math.max(sum[j + 1] - sum[i + 1] - dp[i + 1][j], sum[j] - sum[i] - dp[i][j - 1]);
        }
    }
    return dp[0][n - 1];
};
```

```TypeScript [sol2-TypeScript]
function stoneGameVII(stones: number[]): number {
    const n: number = stones.length;
    const sum: number[] = new Array(n + 1).fill(0);
    const dp: number[][] = new Array(n).fill(0).map(() => new Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + stones[i];
    }
    for (let i = n - 2; i >= 0; i--) {
        for (let j = i + 1; j < n; j++) {
            dp[i][j] = Math.max(sum[j + 1] - sum[i + 1] - dp[i + 1][j], sum[j] - sum[i] - dp[i][j - 1]);
        }
    }

    return dp[0][n - 1];
};
```

**复杂度分析**

- 时间复杂度: $O(n^2)$，其中 $n$ 表示数组的长度。求数组的前缀和需要的时间为 $O(n)$，分别需要求出每个区间 $(i,j)$ 的最优子状态，一共有 $n^2$ 个子状态需要计算，需要的时间为 $O(n^2)$，总共需要的时间为 $O(n^2)$。

- 空间复杂度：$O(n^2)$，其中 $n$ 表示数组的长度。需要存储数组的前缀和，需要的空间为 $O(n)$，需要存储每个索引 $(i,j)$ 对应的最优状态，需要的空间为 $O(n^2)$，总共需要的空间为 $O(n^2)$。