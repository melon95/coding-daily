## 视频讲解

请看[【基础算法精讲 12】](https://www.bilibili.com/video/BV1W44y1Z7AR/)，制作不易，欢迎点赞~

![235.png](https://pic.leetcode.cn/1681701383-TwtQeO-235.png)

```py [sol-Python3]
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        x = root.val
        if p.val < x and q.val < x:  # p 和 q 都在左子树
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > x and q.val > x:  # p 和 q 都在右子树
            return self.lowestCommonAncestor(root.right, p, q)
        return root  # 其它
```

```java [sol-Java]
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        int x = root.val;
        if (p.val < x && q.val < x) { // p 和 q 都在左子树
            return lowestCommonAncestor(root.left, p, q);
        }
        if (p.val > x && q.val > x) { // p 和 q 都在右子树
            return lowestCommonAncestor(root.right, p, q);
        }
        return root; // 其它
    }
}
```

```cpp [sol-C++]
class Solution {
public:
    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
        int x = root->val;
        if (p->val < x && q->val < x) { // p 和 q 都在左子树
            return lowestCommonAncestor(root->left, p, q);
        }
        if (p->val > x && q->val > x) { // p 和 q 都在右子树
            return lowestCommonAncestor(root->right, p, q);
        }
        return root; // 其它
    }
};
```

```go [sol-Go]
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	x := root.Val
	if p.Val < x && q.Val < x { // p 和 q 都在左子树
		return lowestCommonAncestor(root.Left, p, q)
	}
	if p.Val > x && q.Val > x { // p 和 q 都在右子树
		return lowestCommonAncestor(root.Right, p, q)
	}
	return root // 其它
}
```

```js [sol-JavaScript]
var lowestCommonAncestor = function(root, p, q) {
    const x = root.val;
    if (p.val < x && q.val < x) { // p 和 q 都在左子树
        return lowestCommonAncestor(root.left, p, q);
    }
    if (p.val > x && q.val > x) { // p 和 q 都在右子树
        return lowestCommonAncestor(root.right, p, q);
    }
    return root; // 其它
};
```

```rust [sol-Rust]
use std::rc::Rc;
use std::cell::RefCell;

impl Solution {
    pub fn lowest_common_ancestor(root: Option<Rc<RefCell<TreeNode>>>, p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        let x = root.as_ref().unwrap();
        let x_val = x.borrow().val;
        let p_val = p.as_ref().unwrap().borrow().val;
        let q_val = q.as_ref().unwrap().borrow().val;
        if p_val < x_val && q_val < x_val { // p 和 q 都在左子树
            return Self::lowest_common_ancestor(x.borrow_mut().left.take(), p, q);
        }
        if p_val > x_val && q_val > x_val { // p 和 q 都在右子树
            return Self::lowest_common_ancestor(x.borrow_mut().right.take(), p, q);
        }
        root // 其它
    }
}
```

#### 复杂度分析

- 时间复杂度：$\mathcal{O}(n)$，其中 $n$ 为二叉搜索树的节点个数。
- 空间复杂度：$\mathcal{O}(n)$。最坏情况下，二叉搜索树退化成一条链（注意题目没有保证它是**平衡**树），因此递归需要 $\mathcal{O}(n)$ 的栈空间。
