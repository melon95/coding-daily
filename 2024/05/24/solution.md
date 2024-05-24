#### 方法一：贪心 + 单调栈

根据题目对竞争力的定义，我们可以发现越小的数字放置的位置越前，对应的子序列越具竞争力。我们可以用类似单调栈的思想尽量将更小的元素放到子序列的前面，令 $\textit{nums}$ 的大小为 $n$，遍历数组 $\textit{nums}$，假设当前访问的下标为 $i$，对数字 $\textit{nums}[i]$ 执行以下操作：

1. 记栈中的元素数目为 $m$，我们不断地进行操作直到不满足条件：如果 $m > 0$ 且 $m + n - i > k$ 且单调栈的栈顶元素大于 $\textit{nums}[i]$，那么说明栈顶元素可以被当前数字 $\textit{nums}[i]$ 替换，弹出单调栈的栈顶元素。

2. 将 $\textit{nums}[i]$ 压入栈中。

最后返回栈中自下而上的前 $k$ 个元素为结果。

```C++ [sol1-C++]
class Solution {
public:
    vector<int> mostCompetitive(vector<int>& nums, int k) {
        vector<int> res;
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            while (!res.empty() && n - i + res.size() > k && res.back() > nums[i]) {
                res.pop_back();
            }
            res.push_back(nums[i]);
        }
        res.resize(k);
        return res;
    }
};
```

```Java [sol1-Java]
class Solution {
    public int[] mostCompetitive(int[] nums, int k) {
        Deque<Integer> stack = new ArrayDeque<Integer>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && n - i + stack.size() > k && stack.peek() > nums[i]) {
                stack.pop();
            }
            stack.push(nums[i]);
        }
        int[] res = new int[k];
        while (stack.size() > k) {
            stack.pop();
        }
        for (int i = k - 1; i >= 0; i--) {
            res[i] = stack.pop();
        }
        return res;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int[] MostCompetitive(int[] nums, int k) {
        Stack<int> stack = new Stack<int>();
        int n = nums.Length;
        for (int i = 0; i < n; i++) {
            while (stack.Count > 0 && n - i + stack.Count > k && stack.Peek() > nums[i]) {
                stack.Pop();
            }
            stack.Push(nums[i]);
        }
        int[] res = new int[k];
        while (stack.Count > k) {
            stack.Pop();
        }
        for (int i = k - 1; i >= 0; i--) {
            res[i] = stack.Pop();
        }
        return res;
    }
}
```

```Go [sol1-Go]
func mostCompetitive(nums []int, k int) []int {
    res := make([]int, 0, len(nums))
    for i, x := range nums {
        for len(res) > 0 && len(nums) - i + len(res) > k && res[len(res) - 1] > x {
            res = res[:len(res) - 1]
        }
        res = append(res, x)
    }
    return res[:k]
}
```

```Python [sol1-Python3]
class Solution:
    def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
        res = []
        for i, x in enumerate(nums):
            while len(res) > 0 and len(nums) - i + len(res) > k and res[-1] > x:
                res.pop()
            res.append(x)
        return res[:k]
```

```C [sol1-C]
int* mostCompetitive(int* nums, int numsSize, int k, int* returnSize) {
    int n = numsSize;
    int *res = (int *)malloc(sizeof(int) * n);
    *returnSize = k;
    int pos = 0;
    for (int i = 0; i < n; i++) {
        while (pos > 0 && n - i + pos > k && res[pos - 1] > nums[i]) {
            pos--;
        }
        res[pos++] = nums[i];
    }
    return res;
}
```

```JavaScript [sol1-JavaScript]
var mostCompetitive = function(nums, k) {
    const res = [];
    const n = nums.length;
    for (let i = 0; i < n; i++) {
        while (res.length > 0 && n - i + res.length > k && res[res.length - 1] > nums[i]) {
            res.pop();
        }
        res.push(nums[i]);
    }
    res.length = k;
    return res;
};
```

```TypeScript [sol1-TypeScript]
function mostCompetitive(nums: number[], k: number): number[] {
    const res: number[] = [];
    const n = nums.length;
    for (let i = 0; i < n; i++) {
        while (res.length > 0 && n - i + res.length > k && res[res.length - 1] > nums[i]) {
            res.pop();
        }
        res.push(nums[i]);
    }
    res.length = k;
    return res;
};
```

```Rust [sol1-Rust]
impl Solution {
    pub fn most_competitive(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut res = Vec::new();
        let n = nums.len();
        for i in 0..n {
            while res.len() > 0 && (n - i + res.len()) as i32 > k && *res.last().unwrap() > nums[i] {
                res.pop();
            }
            res.push(nums[i]);
        }
        res.truncate(k as usize);
        res
    }
}
```

**复杂度分析**

- 时间复杂度：$O(n)$，其中 $n$ 是数组 $\textit{nums}$ 的大小。

- 空间复杂度：$O(1)$。返回值不计算空间复杂度。