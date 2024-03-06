#### 方法一：枚举每一个二进制位

**思路与算法**

我们直接按照题目描述进行枚举即可。

具体地，我们在外层循环枚举 $i$，内层循环枚举数组 $\textit{nums}$ 中元素 $\textit{nums}[j]$。我们可以通过：

$$
\texttt{(nums[j] >> i) \& 1}
$$

得到 $\textit{nums}[j]$ 的第 $i$ 位。如果至少有 $k$ 个 $1$，就将最终的答案加上 $2^i$，位运算表示即为 $\texttt{1 << i}$。

本题中数组 $\textit{nums}$ 中的元素不超过 $2^{31}$，因此 $i$ 的枚举范围是 $[0, 31)$。

**代码**

```C++ [sol1-C++]
class Solution {
public:
    int findKOr(vector<int>& nums, int k) {
        int ans = 0;
        for (int i = 0; i < 31; ++i) {
            int cnt = 0;
            for (int num: nums) {
                if ((num >> i) & 1) {
                    ++cnt;
                }
            }
            if (cnt >= k) {
                ans |= 1 << i;
            }
        }
        return ans;
    }
};
```

```Java [sol1-Java]
class Solution {
    public int findKOr(int[] nums, int k) {
        int ans = 0;
        for (int i = 0; i < 31; ++i) {
            int cnt = 0;
            for (int num : nums) {
                if (((num >> i) & 1) != 0) {
                    ++cnt;
                }
            }
            if (cnt >= k) {
                ans |= 1 << i;
            }
        }
        return ans;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int FindKOr(int[] nums, int k) {
        int ans = 0;
        for (int i = 0; i < 31; ++i) {
            int cnt = 0;
            foreach (int num in nums) {
                if (((num >> i) & 1) != 0) {
                    ++cnt;
                }
            }
            if (cnt >= k) {
                ans |= 1 << i;
            }
        }
        return ans;
    }
}
```

```Python [sol1-Python3]
class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(31):
            cnt = sum(1 for num in nums if ((num >> i) & 1) > 0)
            if cnt >= k:
                ans |= 1 << i
        return ans
```

```C [sol1-C]
int findKOr(int* nums, int numsSize, int k) {
    int ans = 0;
    for (int i = 0; i < 31; ++i) {
        int cnt = 0;
        for (int j = 0; j < numsSize; j++) {
            int num = nums[j];
            if ((num >> i) & 1) {
                ++cnt;
            }
        }
        if (cnt >= k) {
            ans |= 1 << i;
        }
    }
    return ans;
}
```

```Go [sol1-Go]
func findKOr(nums []int, k int) int {
    ans := 0
    for i := 0; i < 31; i++ {
        cnt := 0
        for _, num := range nums {
            if (num >> i) & 1 == 1 {
                cnt++
            }
        }
        if cnt >= k {
            ans |= 1 << i
        }
    }
    return ans
}
```

```JavaScript [sol1-JavaScript]
var findKOr = function(nums, k) {
    let ans = 0;
    for (let i = 0; i < 31; ++i) {
        let cnt = 0;
        for (const num of nums) {
            if ((num >> i) & 1) {
                ++cnt;
            }
        }
        if (cnt >= k) {
            ans |= 1 << i;
        }
    }
    return ans;
};
```

```TypeScript [sol1-TypeScript]
function findKOr(nums: number[], k: number): number {
    let ans = 0;
    for (let i = 0; i < 31; ++i) {
        let cnt = 0;
        for (const num of nums) {
            if ((num >> i) & 1) {
                ++cnt;
            }
        }
        if (cnt >= k) {
            ans |= 1 << i;
        }
    }
    return ans;
};
```

```Rust [sol1-Rust]
impl Solution {
    pub fn find_k_or(nums: Vec<i32>, k: i32) -> i32 {
        let mut ans = 0;
        for i in 0..31 {
            let mut cnt = 0;
            for &num in nums.iter() {
                if (num >> i) & 1 == 1 {
                    cnt += 1;
                }
            }
            if cnt >= k {
                ans |= 1 << i;
            }
        }
        ans
    }
}
```

**复杂度分析**

- 时间复杂度：$O(n \log C)$，其中 $n$ 是数组 $\textit{nums}$ 的长度，$C$ 是数组 $\textit{nums}$ 中元素的范围。

- 空间复杂度：$O(1)$。