
> Problem: [2641. 二叉树的堂兄弟节点 II](https://leetcode.cn/problems/cousins-in-binary-tree-ii/description/)

[TOC]

# 思路

这道题卡常，深拷贝不行，遍历两遍还是可以。

# 解题方法

采用树的层序遍历，第一次遍历主要做三件事：保存每一层的和到sum数组，数组的下标代表层数；设置孩子节点的fa属性指向父节点；保存孩子节点的总和到父节点的son属性上。

此时，我们获取了三个数据：每一层的总和，每一个节点与其兄弟节点的和（保存在其父节点上），每一个孩子的父亲。显然，将每一层总和减去每个节点与其兄弟节点的和就是堂兄弟节点的和。

第二次遍历主要做一件事：将每个节点和赋值为所在层级的和减去父节点保存的节点和。

# 复杂度

时间复杂度:
$O(n)$

空间复杂度:
$O(n)$



# Code
```JavaScript []
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {TreeNode}
 */
var replaceValueInTree = function(root) {//超时方案
    //先进行层序遍历，保存所有同层节点的和
    let q = new Queue();
    q.enqueue([root, 0]);
    let sum = [];
    while(q.size()){
        let [node, level] = q.dequeue();
        node.son = 0;
        if(sum[level] === undefined){
            sum[level] = node.val;
        }else sum[level] += node.val;
        if(node.left) {
            q.enqueue([node.left, level+1]);
            node.left.fa = node;
            node.son += node.left.val;
        }
        if(node.right) {
            q.enqueue([node.right, level+1]);
            node.right.fa = node;
            node.son += node.right.val;
        }
    }
    //再进行层序遍历，从所有同层节点中删除兄弟节点的数值后保存在当前遍历的节点上
    q.enqueue([root, 0]);
    root.val = 0;
    while(q.size()){
        let [node, level] = q.dequeue();
        if(node.left) {
            q.enqueue([node.left, level+1]);
        }
        if(node.right) {
            q.enqueue([node.right, level+1]);
        }
        if(node.fa)node.val = sum[level] - node.fa.son;
    }
    return root;
};
```
  
