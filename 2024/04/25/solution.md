#### 方法一：模拟

**思路与算法**

直接根据题意进行模拟，主油箱 $\textit{mainTank}$ 每减少 $5$ 个单位，可以行驶 $50$ 千米，此时如果副油箱 $\textit{additionalTank}$ 存在燃料，则从副油箱中减少 $1$ 个单位的燃料累加到主油箱 $\textit{mainTank}$，直到主油箱 $\textit{mainTank}$ 的值小于 $5$ 时，则退出循环，最终历程需要累加 $10 \times \textit{mainTank}$，返回最终结果即可。

**代码**

```C++ [sol1-C++]
class Solution {
public:
    int distanceTraveled(int mainTank, int additionalTank) {
        int ans = 0;
        while (mainTank >= 5) {
            mainTank -= 5;
            ans += 50;
            if (additionalTank > 0) {
                additionalTank--;
                mainTank++;
            }
        }
        return ans + mainTank * 10;
    }
};
```

```C [sol1-C]
int distanceTraveled(int mainTank, int additionalTank){
    int ans = 0;
    while (mainTank >= 5) {
        mainTank -= 5;
        ans += 50;
        if (additionalTank > 0) {
            additionalTank--;
            mainTank++;
        }
    }
    return ans + mainTank * 10;
}
```

```Java [sol1-Java]
class Solution {
    public int distanceTraveled(int mainTank, int additionalTank) {
        int ans = 0;
        while (mainTank >= 5) {
            mainTank -= 5;
            ans += 50;
            if (additionalTank > 0) {
                additionalTank--;
                mainTank++;
            }
        }
        return ans + mainTank * 10;
    }
}
```

```C# [sol1-C#]
public class Solution {
    public int DistanceTraveled(int mainTank, int additionalTank) {
        int ans = 0;
        while (mainTank >= 5) {
            mainTank -= 5;
            ans += 50;
            if (additionalTank > 0) {
                additionalTank--;
                mainTank++;
            }
        }
        return ans + mainTank * 10;
    }
}
```

```Python [sol1-Python3]
class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        ans = 0
        while mainTank >= 5:
            mainTank -= 5
            ans += 50
            if additionalTank > 0:
                additionalTank -= 1
                mainTank += 1
        return ans + mainTank * 10

```

```Go [sol1-Go]
func distanceTraveled(mainTank int, additionalTank int) int {
    ans := 0
    for mainTank >= 5 {
        mainTank -= 5
        ans += 50
        if additionalTank > 0 {
            additionalTank--
            mainTank++
        }
    }
    return ans + mainTank*10
}
```

```JavaScript [sol1-JavaScript]
var distanceTraveled = function(mainTank, additionalTank) {
    let ans = 0;
    while (mainTank >= 5) {
        mainTank -= 5;
        ans += 50;
        if (additionalTank > 0) {
            additionalTank--;
            mainTank++;
        }
    }
    return ans + mainTank * 10;
};
```

```TypeScript [sol1-TypeScript]
function distanceTraveled(mainTank: number, additionalTank: number): number {
    let ans = 0;
    while (mainTank >= 5) {
        mainTank -= 5;
        ans += 50;
        if (additionalTank > 0) {
            additionalTank--;
            mainTank++;
        }
    }
    return ans + mainTank * 10;
};
```

```Rust [sol1-Rust]
impl Solution {
    pub fn distance_traveled(main_tank: i32, additional_tank: i32) -> i32 {
        let mut main_tank = main_tank;
        let mut additional_tank = additional_tank;
        let mut ans = 0;
        while main_tank >= 5 {
            main_tank -= 5;
            ans += 50;
            if additional_tank > 0 {
                additional_tank -= 1;
                main_tank += 1;
            }
        }
        ans + main_tank * 10
    }
}
```

**复杂度分析**

- 时间复杂度：$O(\textit{mainTank})$，$\textit{mainTank}$ 即为给定的主油箱中的初始燃料。实际计算时，最多循环计算 $\lceil \dfrac{\textit{mainTank}}{4} \rceil$ 次。

- 空间复杂度：$O(1)$。

#### 方法二：数学

**思路与算法**

我们仔细分析一下可以知道，假设副油箱中有无限的燃料，则本题等价于经典的「[换水问题](https://leetcode.cn/problems/water-bottles/description/)」，方法二中使用的公式推导过程可以参考「[换水问题](https://leetcode.cn/problems/water-bottles/description/)」，在此不再展开描述。

根据题意知道，由于每次耗费主油箱的 $5$ 升燃料之后则可以从副油箱中得到 $1$ 升燃料，此时根据「[换水问题](https://leetcode.cn/problems/water-bottles/description/)」中的方法二可以知道，最多可以从副油箱中注入的燃料为 $n = \lfloor\dfrac{\textit{mainTank} - 5}{4} + 1\rfloor = \lfloor\dfrac{\textit{mainTank} - 1}{4}\rfloor$。由于此时副油箱中一共有 $\textit{additionalTank}$ 升燃料，意味着最多可以向主油箱**注入** $\textit{additionalTank}$ 升燃料，因此我们取二者的最小值 $\min(\lfloor\dfrac{\textit{mainTank} - 1}{4}\rfloor, \textit{additionalTank})$，主油箱中初始油量为 $\textit{mainTank}$ 升，此时主油箱中总共可以使用的油量即为 $m = \textit{mainTank} + \min(\lfloor\dfrac{\textit{mainTank} - 1}{4}\rfloor, \textit{additionalTank})$，总共可以行驶的里程即为 $10 \times (\textit{mainTank} + \min(\lfloor\dfrac{\textit{mainTank} - 1}{4}\rfloor, \textit{additionalTank}))$。

**代码**

```C++ [sol2-C++]
class Solution {
public:
    int distanceTraveled(int mainTank, int additionalTank) {
        return 10 * (mainTank + min((mainTank - 1) / 4, additionalTank));
    }
};
```

```C [sol2-C]
int distanceTraveled(int mainTank, int additionalTank){
    return 10 * (mainTank + fmin((mainTank - 1) / 4, additionalTank));
}
```

```Java [sol2-Java]
class Solution {
    public int distanceTraveled(int mainTank, int additionalTank) {
        return 10 * (mainTank + Math.min((mainTank - 1) / 4, additionalTank));
    }
}
```

```C# [sol2-C#]
public class Solution {
    public int DistanceTraveled(int mainTank, int additionalTank) {
        return 10 * (mainTank + Math.Min((mainTank - 1) / 4, additionalTank));
    }
}
```

```Python [sol2-Python3]
class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        return 10 * (mainTank + min((mainTank - 1) // 4, additionalTank))
```

```Go [sol2-Go]
func distanceTraveled(mainTank int, additionalTank int) int {
    return 10 * (mainTank + min((mainTank-1)/4, additionalTank))
}
```

```JavaScript [sol2-JavaScript]
var distanceTraveled = function(mainTank, additionalTank) {
    return 10 * (mainTank + Math.min(Math.floor((mainTank - 1) / 4), additionalTank));
};
```

```TypeScript [sol2-TypeScript]
function distanceTraveled(mainTank: number, additionalTank: number): number {
    return 10 * (mainTank + Math.min(Math.floor((mainTank - 1) / 4), additionalTank));
};
```

```Rust [sol2-Rust]
impl Solution {
    pub fn distance_traveled(main_tank: i32, additional_tank: i32) -> i32 {
        10 * (main_tank + additional_tank.min((main_tank - 1) / 4))
    }
}
```

**复杂度分析**

- 时间复杂度：$O(1)$。

- 空间复杂度：$O(1)$。