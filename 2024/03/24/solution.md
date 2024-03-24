


### 方法一：动态规划(完全背包)

我们定义 $f[i][j]$ 表示使用前 $i$ 种硬币，凑出金额 $j$ 的最少硬币数。初始时 $f[0][0] = 0$，其余位置的值均为正无穷。

我们可以枚举使用的最后一枚硬币的数量 $k$，那么有：

$$
f[i][j] = \min(f[i - 1][j], f[i - 1][j - x] + 1, \cdots, f[i - 1][j - k \times x] + k)
$$

其中 $x$ 表示第 $i$ 种硬币的面值。

不妨令 $j = j - x$，那么有：

$$
f[i][j - x] = \min(f[i - 1][j - x], f[i - 1][j - 2 \times x] + 1, \cdots, f[i - 1][j - k \times x] + k - 1)
$$

将二式代入一式，我们可以得到以下状态转移方程：

$$
f[i][j] = \min(f[i - 1][j], f[i][j - x] + 1)
$$

最后答案即为 $f[m][n]$。



```python [sol1-Python3]
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        m, n = len(coins), amount
        f = [[inf] * (n + 1) for _ in range(m + 1)]
        f[0][0] = 0
        for i, x in enumerate(coins, 1):
            for j in range(n + 1):
                f[i][j] = f[i - 1][j]
                if j >= x:
                    f[i][j] = min(f[i][j], f[i][j - x] + 1)
        return -1 if f[m][n] >= inf else f[m][n]
```

```java [sol1-Java]
class Solution {
    public int coinChange(int[] coins, int amount) {
        final int inf = 1 << 30;
        int m = coins.length;
        int n = amount;
        int[][] f = new int[m + 1][n + 1];
        for (var g : f) {
            Arrays.fill(g, inf);
        }
        f[0][0] = 0;
        for (int i = 1; i <= m; ++i) {
            for (int j = 0; j <= n; ++j) {
                f[i][j] = f[i - 1][j];
                if (j >= coins[i - 1]) {
                    f[i][j] = Math.min(f[i][j], f[i][j - coins[i - 1]] + 1);
                }
            }
        }
        return f[m][n] >= inf ? -1 : f[m][n];
    }
}
```

```cpp [sol1-C++]
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int m = coins.size(), n = amount;
        int f[m + 1][n + 1];
        memset(f, 0x3f, sizeof(f));
        f[0][0] = 0;
        for (int i = 1; i <= m; ++i) {
            for (int j = 0; j <= n; ++j) {
                f[i][j] = f[i - 1][j];
                if (j >= coins[i - 1]) {
                    f[i][j] = min(f[i][j], f[i][j - coins[i - 1]] + 1);
                }
            }
        }
        return f[m][n] > n ? -1 : f[m][n];
    }
};
```

```go [sol1-Go]
func coinChange(coins []int, amount int) int {
	m, n := len(coins), amount
	f := make([][]int, m+1)
	const inf = 1 << 30
	for i := range f {
		f[i] = make([]int, n+1)
		for j := range f[i] {
			f[i][j] = inf
		}
	}
	f[0][0] = 0
	for i := 1; i <= m; i++ {
		for j := 0; j <= n; j++ {
			f[i][j] = f[i-1][j]
			if j >= coins[i-1] {
				f[i][j] = min(f[i][j], f[i][j-coins[i-1]]+1)
			}
		}
	}
	if f[m][n] > n {
		return -1
	}
	return f[m][n]
}
```

```ts [sol1-TypeScript]
function coinChange(coins: number[], amount: number): number {
    const m = coins.length;
    const n = amount;
    const f: number[][] = Array(m + 1)
        .fill(0)
        .map(() => Array(n + 1).fill(1 << 30));
    f[0][0] = 0;
    for (let i = 1; i <= m; ++i) {
        for (let j = 0; j <= n; ++j) {
            f[i][j] = f[i - 1][j];
            if (j >= coins[i - 1]) {
                f[i][j] = Math.min(f[i][j], f[i][j - coins[i - 1]] + 1);
            }
        }
    }
    return f[m][n] > n ? -1 : f[m][n];
}
```

```rust [sol1-Rust]
impl Solution {
    pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
        let n = amount as usize;
        let mut f = vec![n + 1; n + 1];
        f[0] = 0;
        for &x in &coins {
            for j in x as usize..=n {
                f[j] = f[j].min(f[j - (x as usize)] + 1);
            }
        }
        if f[n] > n {
            -1
        } else {
            f[n] as i32
        }
    }
}
```

```js [sol1-JavaScript]
/**
 * @param {number[]} coins
 * @param {number} amount
 * @return {number}
 */
var coinChange = function (coins, amount) {
    const m = coins.length;
    const n = amount;
    const f = Array(m + 1)
        .fill(0)
        .map(() => Array(n + 1).fill(1 << 30));
    f[0][0] = 0;
    for (let i = 1; i <= m; ++i) {
        for (let j = 0; j <= n; ++j) {
            f[i][j] = f[i - 1][j];
            if (j >= coins[i - 1]) {
                f[i][j] = Math.min(f[i][j], f[i][j - coins[i - 1]] + 1);
            }
        }
    }
    return f[m][n] > n ? -1 : f[m][n];
};
```

时间复杂度 $O(m \times n)$，空间复杂度 $O(m \times n)$。其中 $m$ 和 $n$ 分别为硬币的种类数和总金额。


我们注意到 $f[i][j]$ 只与 $f[i - 1][j]$ 和 $f[i][j - x]$ 有关，因此我们可以将二维数组优化为一维数组，空间复杂度降为 $O(n)$。


```python [sol2-Python3]
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = amount
        f = [0] + [inf] * n
        for x in coins:
            for j in range(x, n + 1):
                f[j] = min(f[j], f[j - x] + 1)
        return -1 if f[n] >= inf else f[n]
```

```java [sol2-Java]
class Solution {
    public int coinChange(int[] coins, int amount) {
        final int inf = 1 << 30;
        int n = amount;
        int[] f = new int[n + 1];
        Arrays.fill(f, inf);
        f[0] = 0;
        for (int x : coins) {
            for (int j = x; j <= n; ++j) {
                f[j] = Math.min(f[j], f[j - x] + 1);
            }
        }
        return f[n] >= inf ? -1 : f[n];
    }
}
```

```cpp [sol2-C++]
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = amount;
        int f[n + 1];
        memset(f, 0x3f, sizeof(f));
        f[0] = 0;
        for (int x : coins) {
            for (int j = x; j <= n; ++j) {
                f[j] = min(f[j], f[j - x] + 1);
            }
        }
        return f[n] > n ? -1 : f[n];
    }
};
```

```go [sol2-Go]
func coinChange(coins []int, amount int) int {
	n := amount
	f := make([]int, n+1)
	for i := range f {
		f[i] = 1 << 30
	}
	f[0] = 0
	for _, x := range coins {
		for j := x; j <= n; j++ {
			f[j] = min(f[j], f[j-x]+1)
		}
	}
	if f[n] > n {
		return -1
	}
	return f[n]
}
```

```ts [sol2-TypeScript]
function coinChange(coins: number[], amount: number): number {
    const n = amount;
    const f: number[] = Array(n + 1).fill(1 << 30);
    f[0] = 0;
    for (const x of coins) {
        for (let j = x; j <= n; ++j) {
            f[j] = Math.min(f[j], f[j - x] + 1);
        }
    }
    return f[n] > n ? -1 : f[n];
}
```

```js [sol2-JavaScript]
/**
 * @param {number[]} coins
 * @param {number} amount
 * @return {number}
 */
var coinChange = function (coins, amount) {
    const n = amount;
    const f = Array(n + 1).fill(1 << 30);
    f[0] = 0;
    for (const x of coins) {
        for (let j = x; j <= n; ++j) {
            f[j] = Math.min(f[j], f[j - x] + 1);
        }
    }
    return f[n] > n ? -1 : f[n];
};
```


---

有任何问题，欢迎评论区交流，欢迎评论区提供其它解题思路（代码），也可以点个赞支持一下作者哈😄~