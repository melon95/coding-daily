#### 方法一：一次遍历

**思路**

如题意所示，遍历所有元素，统计至少工作了 $\textit{target}$ 的员工数。

**代码**

```C++ [sol1-C++]
class Solution {
public:
    int numberOfEmployeesWhoMetTarget(vector<int>& hours, int target) {
        int ans = 0;
        for (int i = 0; i < hours.size(); i++) {
            if (hours[i] >= target) {
                ans++;
            }
        }
        return ans;
    }
};
```

```Java [sol1-Java]
class Solution {
    public int numberOfEmployeesWhoMetTarget(int[] hours, int target) {
        int ans = 0;
        for (int i = 0; i < hours.length; i++) {
            if (hours[i] >= target) {
                ans++;
            }
        }
        return ans;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int NumberOfEmployeesWhoMetTarget(int[] hours, int target) {
        int ans = 0;
        for (int i = 0; i < hours.Length; i++) {
            if (hours[i] >= target) {
                ans++;
            }
        }
        return ans;
    }
}
```

```Go [sol1-Golang]
func numberOfEmployeesWhoMetTarget(hours []int, target int) int {
    ans := 0
    for _, hour := range hours {
        if hour >= target {
            ans++
        }
    }
    return ans
}
```

```C [sol1-C]
int numberOfEmployeesWhoMetTarget(int* hours, int hoursSize, int target){
    int ans = 0;
    for (int i = 0; i < hoursSize; i++) {
        if (hours[i] >= target) {
            ans++;
        }
    }
    return ans;
}
```

```Python [sol1-Python3]
class Solution(object):
    def numberOfEmployeesWhoMetTarget(self, hours, target):
        ans = 0

        for i in range(0, len(hours)):
            if hours[i] >= target:
                ans += 1
        return ans
```

```JavaScript [sol1-JavaScript]
var numberOfEmployeesWhoMetTarget = function(hours, target) {
    ans = 0

    for (let i = 0; i < hours.length; i++) {
        if (hours[i] >= target) {
            ans++
        }
    }
    return ans
};
```

```TypeScript [sol1-TypeScript]
function numberOfEmployeesWhoMetTarget(hours: number[], target: number): number {
    let ans = 0;

    for (let i = 0; i < hours.length; i++) {
        if (hours[i] >= target) {
            ans++;
        }
    }
    return ans;
};
```

```Rust [sol1-Rust]
impl Solution {
    pub fn number_of_employees_who_met_target(hours: Vec<i32>, target: i32) -> i32 {
        let mut ans = 0;

        for i in 0..hours.len() {
            if hours[i] >= target {
                ans += 1
            }
        }
        return ans;
    }
}
```

**复杂度分析**

- 时间复杂度：$O(n)$，其中 $n$ 为数组的长度。

- 空间复杂度：$O(1)$。