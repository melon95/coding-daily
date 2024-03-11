#### 方法一：按要求遍历

**思路与算法**

我们顺序遍历 $\textit{title}$ 字符串，对于其中每个以空格为分界的单词，我们首先找出它的起始与末尾下标，判断它的长度以进行相应操作：

- 如果长度小于等于 $2$，则我们将该单词全部转化为小写；

- 如果长度大于 $2$，则我们将该单词首字母转化为大写，其余字母转化为小写。

最终，我们将转化后的字符串返回作为答案。

另外，对于 $\texttt{Python}$ 等无法直接对字符串特定字符进行修改的语言，我们可以先将字符串分割为单词，并用数组按顺序储存这些单词。随后，我们逐单词进行上述操作生成新的单词并替换。最后，我们将替换后的单词数组拼接为空格连接的字符串并返回作为答案。

**代码**

```C++ [sol1-C++]
class Solution {
public:
    string capitalizeTitle(string title) {
        int n = title.size();
        int l = 0, r = 0;   // 单词左右边界（左闭右开）
        while (r < n) {
            while (r < n && title[r] != ' ') {
                ++r;
            }
            // 对于每个单词按要求处理
            if (r - l > 2) {
                title[l++] = toupper(title[l]);
            }
            while (l < r) {
                title[l++] = tolower(title[l]);
            }
            l = ++r;
        }
        return title;
    }
};
```

```Java [sol1-Java]
class Solution {
    public String capitalizeTitle(String title) {
        StringBuilder sb = new StringBuilder(title);
        int n = title.length();
        int l = 0, r = 0;   // 单词左右边界（左闭右开）
        while (r < n) {
            while (r < n && sb.charAt(r) != ' ') {
                ++r;
            }
            // 对于每个单词按要求处理
            if (r - l > 2) {
                sb.setCharAt(l, Character.toUpperCase(sb.charAt(l)));
                ++l;
            }
            while (l < r) {
                sb.setCharAt(l, Character.toLowerCase(sb.charAt(l)));
                ++l;
            }
            l = r + 1;
            ++r;
        }
        return sb.toString();
    }
}
```

```C# [sol1-C#]
public class Solution {
    public string CapitalizeTitle(string title) {
        StringBuilder sb = new StringBuilder(title);
        int n = title.Length;
        int l = 0, r = 0;   // 单词左右边界（左闭右开）
        while (r < n) {
            while (r < n && sb[r] != ' ') {
                ++r;
            }
            // 对于每个单词按要求处理
            if (r - l > 2) {
                sb[l] = char.ToUpper(sb[l]);
                ++l;
            }
            while (l < r) {
                sb[l] = char.ToLower(sb[l]);
                ++l;
            }
            l = r + 1;
            ++r;
        }
        return sb.ToString();
    }
}
```

```C [sol1-C]
char* capitalizeTitle(char* title) {
    int n = strlen(title);
    int l = 0, r = 0;   // 单词左右边界（左闭右开）
    while (r < n) {
        while (r < n && title[r] != ' ') {
            ++r;
        }
        // 对于每个单词按要求处理
        if (r - l > 2) {
            title[l++] = toupper(title[l]);
        }
        while (l < r) {
            title[l++] = tolower(title[l]);
        }
        l = ++r;
    }
    return title;
}
```

```Python [sol1-Python3]
class Solution:
    def capitalizeTitle(self, title: str) -> str:
        res = []   # 辅助数组
        for word in title.split():
            # 对于分割的每个单词按要求处理
            if len(word) <= 2:
                res.append(word.lower())
            else:
                res.append(word[0].upper() + word[1:].lower())
        return ' '.join(res)
```

```JavaScript [sol1-JavaScript]
var capitalizeTitle = function(title) {
    const words = title.split(' ');
    let res = new Array();
    for (const word of words) {
        if (word.length > 2) {
            res.push(word[0].toUpperCase() + word.slice(1).toLowerCase());
        } else {
            res.push(word.toLowerCase());
        }
    }
    return res.join(" ");
};
```

```TypeScript [sol1-TypeScript]
function capitalizeTitle(title: string): string {
    const words: string[] = title.split(' ');
    let res: string[] = [];
    for (const word of words) {
        if (word.length > 2) {
            res.push(word[0].toUpperCase() + word.slice(1).toLowerCase());
        } else {
            res.push(word.toLowerCase());
        }
    }
    return res.join(" ");
};
```

```Go [sol1-Go]
func capitalizeTitle(title string) string {
    words := strings.Split(title, " ")
    var res []string
    for _, word := range words {
        if len(word) > 2 {
            res = append(res, strings.ToUpper(string(word[0]))+strings.ToLower(word[1:]))
        } else {
            res = append(res, strings.ToLower(word))
        }
    }
    return strings.Join(res, " ")
}
```

```Rust [sol1-Rust]
impl Solution {
    pub fn capitalize_title(title: String) -> String {
        let words: Vec<&str> = title.split(' ').collect();
        let mut res = Vec::new();
        for word in words {
            if word.len() > 2 {
                res.push(word[..1].to_uppercase() + &word[1..].to_lowercase());
            } else {
                res.push(word.to_lowercase());
            }
        }
        res.join(" ")
    }
}
```


**复杂度分析**

- 时间复杂度：$O(n)$，其中 $n$ 为 $\textit{title}$ 的长度。即为遍历字符串完成操作的时间复杂度。

- 空间复杂度：由于不同语言的字符串相关方法实现有所不同，因此空间复杂度也有所不同：

  -  $\texttt{C++, C}$：$O(1)$。
  -  $\texttt{Java}$ 和 $\texttt{C\#}$：$O(n)$，即为临时字符串的空间开销。
  -  $\texttt{Python,JavaScript, Go, Rust}$：$O(n)$，即为辅助数组的空间开销。