**前序遍历**：按照「根-左子树-右子树」的顺序遍历二叉树。

**中序遍历**：按照「左子树-根-右子树」的顺序遍历二叉树。

我们来看看示例 1 是怎么生成这棵二叉树的。

![lc105-c.png](https://pic.leetcode.cn/1707907886-ICkiSC-lc105-c.png)

**递归边界**：如果 $\textit{preorder}$ 的长度是 $0$，对应着空节点，返回空。

晕递归的同学推荐先看这期视频：[深入理解递归【基础算法精讲 09】](https://www.bilibili.com/video/BV1UD4y1Y769/)

## 写法一

```py [sol-Python3]
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:  # 空节点
            return None
        left_size = inorder.index(preorder[0])  # 左子树的大小
        left = self.buildTree(preorder[1: 1 + left_size], inorder[:left_size])
        right = self.buildTree(preorder[1 + left_size:], inorder[1 + left_size:])
        return TreeNode(preorder[0], left, right)
```

```java [sol-Java]
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        if (n == 0) { // 空节点
            return null;
        }
        int leftSize = indexOf(inorder, preorder[0]); // 左子树的大小
        int[] pre1 = Arrays.copyOfRange(preorder, 1, 1 + leftSize);
        int[] pre2 = Arrays.copyOfRange(preorder, 1 + leftSize, n);
        int[] in1 = Arrays.copyOfRange(inorder, 0, leftSize);
        int[] in2 = Arrays.copyOfRange(inorder, 1 + leftSize, n);
        TreeNode left = buildTree(pre1, in1);
        TreeNode right = buildTree(pre2, in2);
        return new TreeNode(preorder[0], left, right);
    }

    // 返回 x 在 a 中的下标，保证 x 一定在 a 中
    private int indexOf(int[] a, int x) {
        for (int i = 0; ; i++) {
            if (a[i] == x) {
                return i;
            }
        }
    }
}
```

```cpp [sol-C++]
class Solution {
public:
    TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
        if (preorder.empty()) { // 空节点
            return nullptr;
        }
        int left_size = ranges::find(inorder, preorder[0]) - inorder.begin(); // 左子树的大小
        vector<int> pre1(preorder.begin() + 1, preorder.begin() + 1 + left_size);
        vector<int> pre2(preorder.begin() + 1 + left_size, preorder.end());
        vector<int> in1(inorder.begin(), inorder.begin() + left_size);
        vector<int> in2(inorder.begin() + 1 + left_size, inorder.end());
        TreeNode *left = buildTree(pre1, in1);
        TreeNode *right = buildTree(pre2, in2);
        return new TreeNode(preorder[0], left, right);
    }
};
```

```go [sol-Go]
func buildTree(preorder, inorder []int) *TreeNode {
    n := len(preorder)
    if n == 0 { // 空节点
        return nil
    }
    leftSize := slices.Index(inorder, preorder[0]) // 左子树的大小
    left := buildTree(preorder[1:1+leftSize], inorder[:leftSize])
    right := buildTree(preorder[1+leftSize:], inorder[1+leftSize:])
    return &TreeNode{preorder[0], left, right}
}
```

```js [sol-JavaScript]
var buildTree = function(preorder, inorder) {
    const n = preorder.length;
    if (n === 0) { // 空节点
        return null;
    }
    const leftSize = inorder.indexOf(preorder[0]); // 左子树的大小
    const pre1 = preorder.slice(1, 1 + leftSize);
    const pre2 = preorder.slice(1 + leftSize);
    const in1 = inorder.slice(0, leftSize);
    const in2 = inorder.slice(1 + leftSize, n);
    const left = buildTree(pre1, in1);
    const right = buildTree(pre2, in2);
    return new TreeNode(preorder[0], left, right);
};
```

```rust [sol-Rust]
use std::rc::Rc;
use std::cell::RefCell;

impl Solution {
    pub fn build_tree(preorder: Vec<i32>, inorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        if preorder.is_empty() { // 空节点
            return None;
        }
        let left_size = inorder.iter().position(|&x| x == preorder[0]).unwrap(); // 左子树的大小
        let pre1 = preorder[1..1 + left_size].to_vec();
        let pre2 = preorder[1 + left_size..].to_vec();
        let in1 = inorder[..left_size].to_vec();
        let in2 = inorder[1 + left_size..inorder.len()].to_vec();
        let left = Self::build_tree(pre1, in1);
        let right = Self::build_tree(pre2, in2);
        Some(Rc::new(RefCell::new(TreeNode { val: preorder[0], left, right })))
    }
}
```

#### 复杂度分析

- 时间复杂度：$\mathcal{O}(n^2)$，其中 $n$ 为 $\textit{preorder}$ 的长度。最坏情况下二叉树是一条链，我们需要递归 $\mathcal{O}(n)$ 次，每次都需要 $\mathcal{O}(n)$ 的时间查找 $\textit{preorder}[1]$ 和复制数组。
- 空间复杂度：$\mathcal{O}(n^2)$。

## 写法二

上面的写法有两个优化点：

1. 用一个哈希表（或者数组）预处理 $\textit{inorder}$ 每个元素的下标，这样就可以 $\mathcal{O}(1)$ 查到 $\textit{preorder}[0]$ 在 $\textit{inorder}$ 的位置，从而 $\mathcal{O}(1)$ 知道左子树的大小。
2. 把递归参数改成子数组下标区间（**左闭右开区间**）的左右端点，从而避免复制数组。

```py [sol-Python3]
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        index = {x: i for i, x in enumerate(inorder)}

        def dfs(pre_l: int, pre_r: int, in_l: int, in_r: int) -> Optional[TreeNode]:
            if pre_l == pre_r:  # 空节点
                return None
            left_size = index[preorder[pre_l]] - in_l  # 左子树的大小
            left = dfs(pre_l + 1, pre_l + 1 + left_size, in_l, in_l + left_size)
            right = dfs(pre_l + 1 + left_size, pre_r, in_l + 1 + left_size, in_r)
            return TreeNode(preorder[pre_l], left, right)

        return dfs(0, len(preorder), 0, len(inorder))  # 左闭右开区间
```

```java [sol-Java]
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        Map<Integer, Integer> index = new HashMap<>();
        for (int i = 0; i < n; i++) {
            index.put(inorder[i], i);
        }
        return dfs(preorder, 0, n, inorder, 0, n, index); // 左闭右开区间
    }

    private TreeNode dfs(int[] preorder, int preL, int preR, int[] inorder, int inL, int inR, Map<Integer, Integer> index) {
        if (preL == preR) { // 空节点
            return null;
        }
        int leftSize = index.get(preorder[preL]) - inL; // 左子树的大小
        TreeNode left = dfs(preorder, preL + 1, preL + 1 + leftSize, inorder, inL, inL + leftSize, index);
        TreeNode right = dfs(preorder, preL + 1 + leftSize, preR, inorder, inL + 1 + leftSize, inR, index);
        return new TreeNode(preorder[preL], left, right);
    }
}
```

```cpp [sol-C++]
class Solution {
public:
    TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
        int n = preorder.size();
        unordered_map<int, int> index;
        for (int i = 0; i < n; i++) {
            index[inorder[i]] = i;
        }

        function<TreeNode*(int, int, int, int)> dfs = [&](int pre_l, int pre_r, int in_l, int in_r) -> TreeNode* {
            if (pre_l == pre_r) { // 空节点
                return nullptr;
            }
            int left_size = index[preorder[pre_l]] - in_l; // 左子树的大小
            TreeNode *left = dfs(pre_l + 1, pre_l + 1 + left_size, in_l, in_l + left_size);
            TreeNode *right = dfs(pre_l + 1 + left_size, pre_r, in_l + 1 + left_size, in_r);
            return new TreeNode(preorder[pre_l], left, right);
        };
        return dfs(0, n, 0, n); // 左闭右开区间
    }
};
```

```go [sol-Go]
func buildTree(preorder, inorder []int) *TreeNode {
    n := len(preorder)
    index := make(map[int]int, n)
    for i, x := range inorder {
        index[x] = i
    }

    var dfs func(int, int, int, int) *TreeNode
    dfs = func(preL, preR, inL, inR int) *TreeNode {
        if preL == preR { // 空节点
            return nil
        }
        leftSize := index[preorder[preL]] - inL // 左子树的大小
        left := dfs(preL+1, preL+1+leftSize, inL, inL+leftSize)
        right := dfs(preL+1+leftSize, preR, inL+1+leftSize, inR)
        return &TreeNode{preorder[preL], left, right}
    }
    return dfs(0, n, 0, n) // 左闭右开区间
}
```

```js [sol-JavaScript]
var buildTree = function(preorder, inorder) {
    const n = preorder.length;
    const index = new Map();
    for (let i = 0; i < n; i++) {
        index.set(inorder[i], i);
    }

    function dfs(preL, preR, inL, inR) {
        if (preL === preR) { // 空节点
            return null;
        }
        const leftSize = index.get(preorder[preL]) - inL; // 左子树的大小
        const left = dfs(preL + 1, preL + 1 + leftSize, inL, inL + leftSize);
        const right = dfs(preL + 1 + leftSize, preR, inL + 1 + leftSize, inR);
        return new TreeNode(preorder[preL], left, right);
    }
    return dfs(0, n, 0, n); // 左闭右开区间
};
```

```rust [sol-Rust]
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

impl Solution {
    pub fn build_tree(preorder: Vec<i32>, inorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        let n = preorder.len();
        let mut index = HashMap::with_capacity(n);
        for (i, &x) in inorder.iter().enumerate() {
            index.insert(x, i);
        }

        fn dfs(preorder: &Vec<i32>, pre_l: usize, pre_r: usize, inorder: &Vec<i32>, in_l: usize, in_r: usize, index: &HashMap<i32, usize>) -> Option<Rc<RefCell<TreeNode>>> {
            if pre_l == pre_r {
                return None;
            }
            let left_size = index[&preorder[pre_l]] - in_l;
            let left = dfs(preorder, pre_l + 1, pre_l + 1 + left_size, inorder, in_l, in_l + left_size, index);
            let right = dfs(preorder, pre_l + 1 + left_size, pre_r, inorder, in_l + 1 + left_size, in_r, index);
            Some(Rc::new(RefCell::new(TreeNode { val: preorder[pre_l], left, right })))
        }
        dfs(&preorder, 0, n, &inorder, 0, n, &index)
    }
}
```

#### 复杂度分析

- 时间复杂度：$\mathcal{O}(n)$，其中 $n$ 为 $\textit{preorder}$ 的长度。递归 $\mathcal{O}(n)$ 次，每次只需要 $\mathcal{O}(1)$ 的时间。
- 空间复杂度：$\mathcal{O}(n)$。

> 注：由于哈希表常数比数组大，实际运行效率可能不如写法一。

## 构造系列

这三题都可以用本文讲的套路解决。

- [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
- [889. 根据前序和后序遍历构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

欢迎关注 [B站@灵茶山艾府](https://space.bilibili.com/206214)

更多精彩题解，请看 [往期题解精选（已分类）](https://github.com/EndlessCheng/codeforces-go/blob/master/leetcode/SOLUTIONS.md)
