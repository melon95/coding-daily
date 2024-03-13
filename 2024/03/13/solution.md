#### 方法一：贪心

**思路与算法**

题目给定**二进制**字符串 $s$ 构造字典序最大的**二进制奇数**，根据定义可以知道字符串中每一位要么为 $0$，要么为 $1$。由于构造的数必须为奇数，则最低位必须为 $1$，因此我们从字符串 $s$ 中选择一个 $1$ 放置到最低位。按照贪心原则，其余的 $1$ 全部放在最高位，剩余的 $0$ 放在剩下的位即可，直接构造目标字符串返回即可。

**代码**

```C++ [sol1-C++]
class Solution {
public:
    string maximumOddBinaryNumber(string s) {
        int cnt = count(s.begin(), s.end(), '1');
        return string(cnt - 1, '1') + string(s.length() - cnt, '0') + '1';
    }
};
```

```Java [sol1-Java]
class Solution {
    public String maximumOddBinaryNumber(String s) {
        int cnt = 0;
        for (int i = 0; i < s.length(); i++) {
            cnt += s.charAt(i) - '0';
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cnt - 1; i++) {
            sb.append('1');
        }
        for (int i = 0; i < s.length() - cnt; i++) {
            sb.append('0');
        }
        sb.append('1');
        return sb.toString();
    }
}
```

```C# [sol1-C#]
public class Solution {
    public string MaximumOddBinaryNumber(string s) {
        int cnt = 0;
        foreach (char c in s) {
            cnt += c - '0';
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cnt - 1; i++) {
            sb.Append('1');
        }
        for (int i = 0; i < s.Length - cnt; i++) {
            sb.Append('0');
        }
        sb.Append('1');
        return sb.ToString();
    }
}
```

```C [sol1-C]
char* maximumOddBinaryNumber(char* s){
    int cnt = 0, len = strlen(s);
    for (int i = 0; s[i] != '\0'; i++) {
        if (s[i] == '1') {
            cnt++;
        }
    }
    char *res = (char *)malloc(sizeof(char) * (len + 1));
    for (int pos = 0; pos < len - 1; pos++) {
        if (pos < cnt - 1) {
            res[pos] = '1';
        } else {
            res[pos] = '0';
        }
    }
    res[len - 1] = '1';
    res[len] = '\0';
    return res;
}
```

```Python [sol1-Python3]
class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        cnt = s.count('1')
        return '1' * (cnt - 1) + '0' * (len(s) - cnt) + '1'
```

```Go [sol1-Go]
func maximumOddBinaryNumber(s string) string {
    cnt := strings.Count(s, "1")
	return strings.Repeat("1", cnt - 1) + strings.Repeat("0", len(s) - cnt) + "1"
}
```

```JavaScript [sol1-JavaScript]
var maximumOddBinaryNumber = function(s) {
    let cnt = 0;
    for (let i = 0; i < s.length; i++) {
        if (s[i] == '1') {
            cnt++;
        }
    }
    return '1'.repeat(cnt - 1) + '0'.repeat(s.length - cnt) + '1';
};
```

```TypeScript [sol1-TypeScript]
function maximumOddBinaryNumber(s: string): string {
    let cnt = 0;
    for (let i = 0; i < s.length; i++) {
        if (s[i] == '1') {
            cnt++;
        }
    }
    return '1'.repeat(cnt - 1) + '0'.repeat(s.length - cnt) + '1';
};
```

```Rust [sol1-Rust]
impl Solution {
    pub fn maximum_odd_binary_number(s: String) -> String {
        let cnt = s.chars().filter(|&c| c == '1').count();
        "1".repeat(cnt - 1) + &*"0".repeat(s.len() - cnt) + "1"
    }
}
```

**复杂度分析**

- 时间复杂度：$O(n)$，其中 $n$ 表示给定字符串的长度。只需要遍历一遍字符串即可。

- 空间复杂度：$O(1)$，除返回值外不需要额外的空间。