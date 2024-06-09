### 解题思路
- 运用动态规划算法的一个重要条件：**子问题必须独立**。
- 而当前题目中我们每戳破一个气球nums[i]，得到的分数和该气球相邻的气球nums[i-1]和nums[i+1]是有相关性的。
- 如果想要用动态规划，必须巧妙地定义dp数组的含义，避免子问题产生相关性，才能推出合理的状态转移方程。
- 题目说可以认为nums[-1] = nums[n] = 1，那么我们先直接把这两个边界加进去，形成一个新的数组points
```
  let n = nums.length;
  // 添加两侧的虚拟气球
  let points = new Array(n + 2);
  points[0] = points[n + 1] = 1;
  for (let i = 1; i <= n; i++) {
    points[i] = nums[i - 1];
  }
```
- 所以其实题目求的是**在一排气球points中，请你戳破气球0和气球n+1之间的所有气球（不包括0和n+1），使得最终只剩下气球0和气球n+1两个气球，最多能够得到多少分？**

- 定义dp数组的含义：dp[i][j] = x表示，戳破气球i和气球j之间（开区间，不包括i和j）的所有气球，可以获得的最高分数为x。
- 那么根据这个定义，题目要求的结果就是dp[0][n+1]的值
- base case 就是dp[i][j] = 0，其中0 <= i <= n+1, j <= i+1，因为这种情况下，开区间(i, j)中间根本没有气球可以戳。
- 想一想气球i和气球j之间最后一个被戳破的气球可能是哪一个？
- 其实气球i和气球j之间的所有气球都可能是最后被戳破的那一个，不防假设为k。
- i和j就是两个「状态」，最后戳破的那个气球k就是「选择」。
- 根据对dp数组的定义，如果最后一个戳破气球k，dp[i][j]的值应该为：
```
dp[i][j] = dp[i][k] + dp[k][j] 
         + points[i]*points[k]*points[j]
```
- 也就是说最后戳破的是气球k。那得先把开区间(i, k)的气球都戳破，再把开区间(k, j)的气球都戳破；最后剩下的气球k，相邻的就是气球i和气球j，这时候戳破k的话得到的分数就是points[i]*points[k]*points[j]。
- 戳破开区间(i, k)和开区间(k, j)的气球最多能得到的分数是多少呢？嘿嘿，就是dp[i][k]和dp[k][j]，这恰好就是我们对dp数组的定义
![image.png](https://pic.leetcode-cn.com/1642319931-eOaoRW-image.png)
- 关于「状态」的穷举，最重要的一点就是：**状态转移所依赖的状态必须被提前计算出来**。
- dp[i][j]所依赖的状态是dp[i][k]和dp[k][j]，那么我们必须保证：在计算dp[i][j]时，dp[i][k]和dp[k][j]已经被计算出来了（其中i < k < j）
- **根据 base case 和最终状态进行推导**
- 先把 base case 和最终的状态在 DP table 上画出来
![image.png](https://pic.leetcode-cn.com/1642320078-gOabhE-image.png)
- 对于任一dp[i][j]，我们希望所有dp[i][k]和dp[k][j]已经被计算，画在图上就是下面这种情况：
![image.png](https://pic.leetcode-cn.com/1642320159-WLjLgi-image.png)

- 为了达到这个要求，可以有两种遍历方法，要么斜着遍历，要么从下到上从左到右遍历：
![image.png](https://pic.leetcode-cn.com/1642320218-DZDEiU-image.png)


![image.png](https://pic.leetcode-cn.com/1642320234-rvNYoh-image.png)
- 题解中给出的是从下往上遍历的实现

- 参考文献 [经典动态规划：戳气球问题](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485172&idx=1&sn=b860476b205b04f960ea0de6f70d3553&scene=21#wechat_redirect)

### 代码

```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
var maxCoins = function (nums) {
  let n = nums.length;
  // 添加两侧的虚拟气球
  let points = new Array(n + 2);
  points[0] = points[n + 1] = 1;
  for (let i = 1; i <= n; i++) {
    points[i] = nums[i - 1];
  }
  // base case 已经都被初始化为 0
  let dp = new Array(n + 2).fill(0).map(() => new Array(n + 2).fill(0));
  // 开始状态转移
  // i 应该从下往上
  for (let i = n; i >= 0; i--) {
    // j 应该从左往右
    for (let j = i + 1; j < n + 2; j++) {
      // 最后戳破的气球是哪个？
      for (let k = i + 1; k < j; k++) {
        // 择优做选择
        dp[i][j] = Math.max(
          dp[i][j],
          dp[i][k] + dp[k][j] + points[i] * points[j] * points[k]
        );
      }
    }
  }
  return dp[0][n + 1];
};
```