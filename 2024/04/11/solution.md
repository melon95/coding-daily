#### 方法一：深度优先搜索

很容易想到，对于树中的一个节点可以采用深度优先搜索，由于根节点到任意节点的搜索路径就是该节点的祖先节点的集合，所以搜索路径上最近的与其互质的就是答案。在搜索遍历树处理答案的时候，有许多信息可以复用。

从数据范围入手，发现 $\textit{nums}[i]$ 的范围是 $[1,50]$，预处理 $\textit{gcds}[j]$ 表示 $[1,50]$ 中与 $j$ 互质的元素的集合。用 $\textit{tmp}[i]$ 表示在搜索过程中 $i = \textit{nums}[x]$ 的节点坐标集合，显然该集合的末尾元素离当前节点最近。对于任意一个节点 $x$，初始化 $\textit{ans}[x] = -1$，通过预处理的 $\textit{gcds}$ 数组得到与 $\textit{nums}[x]$ 互质的所有整数。对于每个整数 $y$，$\textit{tmp}[y]$ 中存储的最后一个位置，就是与当前节点距离最近的祖先节点的位置。找到所有这样的节点，并与当前节点 $\textit{ans}[x]$ 的深度进行比较后更新。

需要注意一下，题目中的边并没有指定方向。同时当前节点 $x$ 在搜索子节点前，令 $i = \textit{nums}[x]$，会放进 $\textit{tmp}[i]$ 的末尾，在回溯时需要 $\textit{pop}$ 出来。

```C++ [sol1-C++]
class Solution {
private:
    vector<vector<int>> gcds; 
    vector<vector<int>> tmp;
    vector<vector<int>> g;
    vector<int> dep;
    vector<int> ans;

public:
    void dfs(vector<int> &nums, int x, int depth) {
        dep[x] = depth;
        for (int val : gcds[nums[x]]) {
            if (tmp[val].empty()) {
                continue;
            }
        
            int las = tmp[val].back();
            if (ans[x] == -1 || dep[las] > dep[ans[x]]) {
                ans[x] = las;
            }
        }
        tmp[nums[x]].push_back(x);

        for (int val : g[x]) {
            if (dep[val] == -1) { // 被访问过的点dep不为-1
                dfs(nums, val, depth + 1);
            }
        }

        tmp[nums[x]].pop_back();
    }

    vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& edges) {
        int n = nums.size();
        
        // 初始化
        gcds.resize(51);
        tmp.resize(51);
        ans.resize(n, -1);
        dep.resize(n, -1);
        g.resize(n);

        for (int i = 1; i <= 50; i++) {
            for (int j = 1; j <= 50; j++) {
                if (gcd(i, j) == 1) {
                    gcds[i].push_back(j);
                } 
            }
        }
        
        for (const auto &val : edges) {
            g[val[0]].push_back(val[1]);
            g[val[1]].push_back(val[0]);
        }

        dfs(nums, 0, 1);
        
        return ans;
    }
};
```

```Java [sol1-Java]
class Solution {
    List<Integer>[] gcds;
    List<Integer>[] tmp;
    List<Integer>[] g;
    int[] dep;
    int[] ans;

    public int[] getCoprimes(int[] nums, int[][] edges) {
        int n = nums.length;
        
        // 初始化
        gcds = new List[51];
        tmp = new List[51];
        for (int i = 0; i <= 50; i++) {
            gcds[i] = new ArrayList<Integer>();
            tmp[i] = new ArrayList<Integer>();
        }
        ans = new int[n];
        dep = new int[n];
        Arrays.fill(ans, -1);
        Arrays.fill(dep, -1);
        g = new List[n];
        for (int i = 0; i < n; i++) {
            g[i] = new ArrayList<Integer>();
        }

        for (int i = 1; i <= 50; i++) {
            for (int j = 1; j <= 50; j++) {
                if (gcd(i, j) == 1) {
                    gcds[i].add(j);
                } 
            }
        }

        for (int[] val : edges) {
            g[val[0]].add(val[1]);
            g[val[1]].add(val[0]);
        }

        dfs(nums, 0, 1);
        
        return ans;
    }

    public int gcd(int x, int y) {
        while (y != 0) {
            int temp = x;
            x = y;
            y = temp % y;
        }
        return x;
    }

    public void dfs(int[] nums, int x, int depth) {
        dep[x] = depth;
        for (int val : gcds[nums[x]]) {
            if (tmp[val].isEmpty()) {
                continue;
            }
        
            int las = tmp[val].get(tmp[val].size() - 1);
            if (ans[x] == -1 || dep[las] > dep[ans[x]]) {
                ans[x] = las;
            }
        }
        tmp[nums[x]].add(x);

        for (int val : g[x]) {
            if (dep[val] == -1) { // 被访问过的点dep不为-1
                dfs(nums, val, depth + 1);
            }
        }

        tmp[nums[x]].remove(tmp[nums[x]].size() - 1);
    }
}
```

```C [sol1-C]
struct ListNode *createListNode(int val) {
    struct ListNode *obj = (struct ListNode *)malloc(sizeof(struct ListNode));
    obj->val = val;
    obj->next = NULL;
    return obj;
}

void freeList(struct ListNode *list) {
    while (list) {
        struct ListNode *p = list;
        list = list->next;
        free(p);
    }
}

void dfs(int x, int depth, int *dep, int *ans, const int *nums, struct ListNode **tmp, struct ListNode **gcds, struct ListNode **g) {
    dep[x] = depth;
    for (struct ListNode *p = gcds[nums[x]]; p; p = p->next) {
        int val = p->val;
        if (tmp[val] == NULL) {
            continue;
        }
        int las = tmp[val]->val;
        if (ans[x] == -1 || dep[las] > dep[ans[x]]) {
            ans[x] = las;
        }
    }
    struct ListNode *node = createListNode(x);
    node->next = tmp[nums[x]];
    tmp[nums[x]] = node;
    for (struct ListNode *p = g[x]; p; p = p->next) {
        int val = p->val;
        if (dep[val] == -1) { // 被访问过的点dep不为-1
            dfs(val, depth + 1, dep, ans, nums, tmp, gcds, g);
        }
    }
    node = tmp[nums[x]];
    tmp[nums[x]] = tmp[nums[x]]->next;
    free(node);
}

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int* getCoprimes(int* nums, int numsSize, int** edges, int edgesSize, int* edgesColSize, int* returnSize) {
    int n = numsSize;

    struct ListNode *gcds[51];
    struct ListNode *tmp[51];
    struct ListNode *g[n];
    int *ans = (int *)malloc(sizeof(int) * n);
    int *dep = (int *)malloc(sizeof(int) * n);
    // 初始化
    for (int i = 0; i < n; i++) {
        g[i] = NULL;
        ans[i] = -1;
        dep[i] = -1;
    }
    for (int i = 1; i <= 50; i++) {
        gcds[i] = NULL;
        tmp[i] = NULL;
        for (int j = 1; j <= 50; j++) {
            if (gcd(i, j) == 1) {
                struct ListNode *p = createListNode(j);
                p->next = gcds[i];
                gcds[i] = p;
            } 
        }
    }
    for (int i = 0; i < edgesSize; i++) {
        int x = edges[i][0], y = edges[i][1];
        struct ListNode *nodex = createListNode(x);
        nodex->next = g[y];
        g[y] = nodex;
        struct ListNode *nodey = createListNode(y);
        nodey->next = g[x];
        g[x] = nodey;
    }

    dfs(0, 1, dep, ans, nums, tmp, gcds, g);
    *returnSize = n;
    for (int i = 1; i <= 50; i++) {
        freeList(gcds[i]);
        freeList(tmp[i]);
    }
    for (int i = 0; i < n; i++) {
        freeList(g[i]);
    }
    free(dep);
    return ans;
}
```

```Python [sol1-Python3]
class Solution:
    def getCoprimes(self, nums: List[int], edges: List[List[int]]) -> List[int]:
        n = len(nums)
        # 初始化
        gcds = [[] for _ in range(51)]
        tmp = [[] for _ in range(51)]
        ans = [-1] * n
        dep = [-1] * n 
        g = [[] for _ in range(n)]

        def dfs(x: int, depth: int):
            dep[x] = depth
            for val in gcds[nums[x]]:
                if not tmp[val]:
                    continue
                las = tmp[val][-1]
                if ans[x] == -1 or dep[las] > dep[ans[x]]:
                    ans[x] = las
            tmp[nums[x]].append(x)
            for val in g[x]:
                if dep[val] == -1: # 被访问过的点dep不为-1
                    dfs(val, depth + 1)
            tmp[nums[x]].pop()

        for i in range(1, 51):
            for j in range(1, 51):
                if math.gcd(i, j) == 1:
                    gcds[i].append(j)
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        dfs(0, 1)
        return ans
```

```Go [sol1-Go]
func getCoprimes(nums []int, edges [][]int) []int {
    n := len(nums)
	gcds := make([][]int, 51)
	tmp := make([][]int, 51)
	ans := make([]int, n)
	dep := make([]int, n)
	g := make([][]int, n)
    // 初始化
	for i := 0; i < 51; i++ {
		gcds[i] = []int{}
		tmp[i] = []int{}
	}
	for i := 0; i < n; i++ {
		g[i] = []int{}
        ans[i], dep[i] = -1, -1
	}

	var dfs func(x, depth int)
	dfs = func(x, depth int) {
		dep[x] = depth
		for _, val := range gcds[nums[x]] {
			if len(tmp[val]) == 0 {
				continue
			}
			las := tmp[val][len(tmp[val]) - 1]
			if ans[x] == -1 || dep[las] > dep[ans[x]] {
				ans[x] = las
			}
		}
		tmp[nums[x]] = append(tmp[nums[x]], x)
		for _, val := range g[x] {
			if dep[val] == -1 { // 被访问过的点dep不为-1
				dfs(val, depth + 1)
			}
		}
		tmp[nums[x]] = tmp[nums[x]][:len(tmp[nums[x]]) - 1]
	}

	for i := 1; i <= 50; i++ {
		for j := 1; j <= 50; j++ {
			if gcd(i, j) == 1 {
				gcds[i] = append(gcds[i], j)
			}
		}
	}
	for _, edge := range edges {
		x := edge[0]
		y := edge[1]
		g[x] = append(g[x], y)
		g[y] = append(g[y], x)
	}

	dfs(0, 1)
	return ans
}

func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a % b
    }
    return a
}
```

```JavaScript [sol1-JavaScript]
var getCoprimes = function(nums, edges) {
    const n = nums.length;
    const gcds = Array.from({ length: 51 }, () => []);
    const tmp = Array.from({ length: 51 }, () => []);
    const ans = Array(n).fill(-1);
    const dep = Array(n).fill(-1);
    const g = Array.from({ length: n }, () => []);

    function gcd(a, b) {
        while (b !== 0) {
            [a, b] = [b, a % b];
        }
        return a;
    }

    function dfs(x, depth) {
        dep[x] = depth;
        for (const val of gcds[nums[x]]) {
            if (tmp[val].length === 0) continue;
            const las = tmp[val][tmp[val].length - 1];
            if (ans[x] === -1 || dep[las] > dep[ans[x]]) {
                ans[x] = las;
            }
        }
        tmp[nums[x]].push(x);
        for (const val of g[x]) {
            if (dep[val] === -1) { // 被访问过的点dep不为-1
                dfs(val, depth + 1);
            }
        }
        tmp[nums[x]].pop();
    }

    // 初始化
    for (let i = 1; i <= 50; i++) {
        for (let j = 1; j <= 50; j++) {
            if (gcd(i, j) === 1) {
                gcds[i].push(j);
            }
        }
    }
    for (const [x, y] of edges) {
        g[x].push(y);
        g[y].push(x);
    }

    dfs(0, 1);
    return ans;
};
```

```TypeScript [sol1-TypeScript]
function getCoprimes(nums: number[], edges: number[][]): number[] {
    const n: number = nums.length;
    const gcds: number[][] = Array.from({ length: 51 }, () => []);
    const tmp: number[][] = Array.from({ length: 51 }, () => []);
    const ans: number[] = Array(n).fill(-1);
    const dep: number[] = Array(n).fill(-1);
    const g: number[][] = Array.from({ length: n }, () => []);

    function gcd(a: number, b: number): number {
        while (b !== 0) {
            [a, b] = [b, a % b];
        }
        return a;
    }

    function dfs(x: number, depth: number): void {
        dep[x] = depth;
        for (const val of gcds[nums[x]]) {
            if (tmp[val].length === 0) continue;
            const las: number = tmp[val][tmp[val].length - 1];
            if (ans[x] === -1 || dep[las] > dep[ans[x]]) {
                ans[x] = las;
            }
        }
        tmp[nums[x]].push(x);
        for (const val of g[x]) {
            if (dep[val] === -1) { // 被访问过的点dep不为-1
                dfs(val, depth + 1);
            }
        }
        tmp[nums[x]].pop();
    }

    // 初始化
    for (let i = 1; i <= 50; i++) {
        for (let j = 1; j <= 50; j++) {
            if (gcd(i, j) === 1) {
                gcds[i].push(j);
            }
        }
    }
    for (const [x, y] of edges) {
        g[x].push(y);
        g[y].push(x);
    }

    dfs(0, 1);
    return ans;
};
```

```Rust [sol1-Rust]
impl Solution {
    pub fn get_coprimes(nums: Vec<i32>, edges: Vec<Vec<i32>>) -> Vec<i32> {
        let n = nums.len();
        let mut gcds = vec![vec![]; 51];
        let mut tmp = vec![vec![]; 51];
        let mut ans = vec![-1; n];
        let mut dep = vec![-1; n];
        let mut g = vec![vec![]; n];

        fn gcd(a: i32, b: i32) -> i32 {
            let mut a = a;
            let mut b = b;
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a.abs()
        }

        fn dfs(x: usize, depth: i32, dep: &mut [i32], ans: &mut [i32], nums: &Vec<i32>, tmp: &mut Vec<Vec<usize>>, gcds: &Vec<Vec<i32>>, g: &Vec<Vec<usize>>) {
            dep[x] = depth;
            for &val in &gcds[nums[x] as usize] {
                if tmp[val as usize].is_empty() {
                    continue;
                }
                let las = *tmp[val as usize].last().unwrap();
                if ans[x] == -1 || dep[las] > dep[ans[x] as usize] {
                    ans[x] = las as i32;
                }
            }
            tmp[nums[x] as usize].push(x);
            for &val in &g[x] {
                if dep[val] == -1 { // 被访问过的点dep不为-1
                    dfs(val, depth + 1, dep, ans, nums, tmp, gcds, g);
                }
            }
            tmp[nums[x] as usize].pop();
        }

        // 初始化
        for i in 1..= 50 {
            for j in 1..= 50 {
                if gcd(i, j) == 1 {
                    gcds[i as usize].push(j);
                }
            }
        }
        for edge in &edges {
            let x = edge[0] as usize;
            let y = edge[1] as usize;
            g[x].push(y);
            g[y].push(x);
        }
        dfs(0, 1, &mut dep, &mut ans, &nums, &mut tmp, &gcds, &g);
        ans
    }
}
```

**复杂度分析**

- 时间复杂度：$O(C \times n)$。$n$ 为数组 $\textit{nums}$ 的长度，$C$ 表示 $\textit{nums}$ 的数据范围，在本题中最大为 $50$。

- 空间复杂度：$O(C^2 \log C + C \times n)$。