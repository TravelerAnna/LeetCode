#  Leetcode

## Hackerank 

### 1. Python

#### Print consecutive numbers as string 

Given number n, print 123```n, without string method.

```python
result = []
for i in range(1, n+1):
    result.append(i)
for num in result:
    print(num, end='')
```



#### Print numbers in decimals

*print ( f " { num_posi : .6f } " )*             # six decimals



#### Time Conversion

Convert 12-h clock to 24-h clock.

```python
def timeConversion(s):
    str_s = re.split(':', s)
    if ('PM' in str_s[-1]):
        hour = int(str_s[0])
        second = re.split('PM', str_s[-1])
        if (hour != 12):
            hour = hour + 12
            str_s[0] = str(hour)
        str_s[-1] = second[0]
    if ('AM' in str_s[-1]):
        hour = int(str_s[0])
        second = re.split('AM', str_s[-1])
        if (hour == 12):
            hour = hour - 12
            str_s[0] = '0' + str(hour)
        str_s[-1] = second[0]
    return ':'.join(str_s)
```

1. *re.split( ':', s)* : split string in specified character, to an list

2. Convert string to number: *int(string), float(string)*

3. Join string : *':'.join(str_s)*







## Time Complexity

**<u>O(1)常数阶 < O(logn)对数阶 < O(n)线性阶 < O(nlogn)线性对数阶 < O(n^2)平方阶 < O(n^3)立方阶 < O(2^n)指数阶</u>**

These ignores the constant coefficient and the data scale is very large.



#### 1. What is time complexity?

<u>Time Complexity is a funtion to desribe the running time of an algorithm.</u>

To estimate the program running time, we usually estimate the number of operation units of the algorithm to represent the time consumed by the program. 

Here, it is assumed that <u>the time consumed by each unit of the CPU is the same.</u>

Assuming that the problem scale of the algorithm is `n`, the number of operation units is represented by the function `f(n)`. 

<u>As the data scale n increases, the growth rate of the algorithm execution time is the same as the growth rate of f(n).</u> 

This is called the asymptotic time complexity of the algorithm, or time complexity for short, denoted as **O(f(n))**.



#### 2. What is O?

Big O is used to express an upper bound. 

When it is used as the upper bound of the worst-case running time of an algorithm, it is the upper bound of the running time for any data input.

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20241102203414976.png" alt="image-20241102203414976" style="zoom:200%;" />

<img src="https://camo.githubusercontent.com/a80cb0b468b7373e90684da36a05f7c513944d0cc1ffd404720f8b7d9fd4dae5/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303732383138353734353631312d32303233303331303132333834343330362e706e67" alt="时间复杂度4，一般情况下的时间复杂度" style="zoom: 45%;" />

The time complexity of an algorithm mentioned in an interview is the genaral case.

 However, if the interviewer discusses the implementation and performance of an algorithm with us in depth, we must always keep in mind that the <u>time complexity is different for different data use cases.</u>



#### 3. Different Data Size

<img src="https://camo.githubusercontent.com/a5c6669595a60f41623770c12ad02cb88e0540e2de6d0f414a333a699496c366/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303732383139313434373338342d32303233303331303132343031353332342e706e67" alt="时间复杂度，不同数据规模的差异" style="zoom:67%;" />

It is not the case that the lower the time complexity, the better we get. 

A. The simplified time complexity ignores constant terms. 

B. The data scale must be considered. If the data scale is very small, even an O(n^2) algorithm may be more appropriate than an O(n) algorithm (when there are constant terms).

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20241102205415933.png" alt="image-20241102205415933" style="zoom:200%;" />

<img src="https://camo.githubusercontent.com/ba62746bf17afec55cd4f061bc2f5af1004b04a664f43173cced131bef6a2d88/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230313230383233313535393137352d32303233303331303132343332353135322e706e67" alt="程序超时1" style="zoom:33%;" />



#### 4. Simplify time complexity function

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20241102213522066.png" alt="image-20241102213522066" style="zoom:200%;" />



#### 5. What's the base in O(log n)?

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20241102213741212.png" alt="image-20241102213741212" style="zoom:200%;" />



#### 6. Example

(1)

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20241102215425071.png" alt="image-20241102215425071" style="zoom:200%;" />



（2）Time Complexity of Recursion algorithm

Calculate x to the power of n

```c++
int function1(int x, int n) {
    int result = 1;  // 注意 任何数的0次方等于1
    for (int i = 0; i < n; i++) {
        result = result * x;
    }
    return result;
}
```

The time complexity is O(n). 

At this time, the interviewer will ask, is there a more efficient algorithm?

If you have no idea at this time, don't say: I don't know, etc.

You can discuss with the interviewer and ask: "Can you give me some hints?" The interviewer prompts: "Consider a recursive algorithm."

```c++
int function2(int x, int n) {
    if (n == 0) {
        return 1; // return 1 同样是因为0次方是等于1的
    }
    return function2(x, n - 1) * x;
}
```

Some students may think of O(log n) when they see recursion, but this is not the case. The time complexity of a recursive algorithm essentially depends on: the number of recursions * the number of operations in each recursion.

Each time n-1, the time complexity of n recursions is O(n), and each time a multiplication operation is performed. The time complexity of the multiplication operation is a constant term O(1), so the time complexity of this code is n × 1 = O(n).

This time complexity did not meet the interviewer's expectations. So I wrote the following recursive algorithm code:

```c++
int function3(int x, int n) {
    if (n == 0) return 1;
    if (n == 1) return x;

    if (n % 2 == 1) {
        return function3(x, n / 2) * function3(x, n / 2)*x;
    }
    return function3(x, n / 2) * function3(x, n / 2);
}

int function4(int x, int n) {
    if (n == 0) return 1;
    if (n == 1) return x;
    int t = function4(x, n / 2);// 这里相对于function3，是把这个递归操作抽取出来
    if (n % 2 == 1) {
        return t * t * x;
    }
    return t * t;
}
```

There is only one recursive call in the last function, and each time it is n/2, so here we call the logarithm of n with the base of 2 a total of times.

Each recursive call is a multiplication operation, which is also a constant operation, so the time complexity of this recursive algorithm is truly O(logn).

![image-20241103215735290](/Users/annahuang/Library/Application Support/typora-user-images/image-20241103215735290.png)





## 1. Arrays

An array is a collection of data of the **same type** stored in **contiguous memory** space.

<img src="https://camo.githubusercontent.com/44ac0154ced4c18194937d6ce191f3940ead4569c326dab8daf6d3b04d737a36/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f706963732f2545372541452539372545362542332539352545392538302539412545352538352542332545362539352542302545372542422538342e706e67" alt="算法通关数组" style="zoom:67%;" />

#### **Summary**

**==Loop Invariant Principle==**

![image-20241208210010420](/Users/annahuang/Library/Application Support/typora-user-images/image-20241208210010420.png)





#### 704. Binary Search

Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.

You must write an algorithm with `O(log n)` runtime complexity.

https://leetcode.cn/problems/binary-search/



- Prerequisites for using the binary search method.  **O(log n)**

**<u>The array is an ordered array, and there are no duplicate elements in the array.</u>** Because once there are duplicate elements, the element subscript returned by the binary search method may not be unique. 



- Method:

1. Ensure your interval, left and right side, **middle(left + distance/2)**.

2. There are generally two definitions of intervals: left closed and right closed, i.e. [left, right], or left closed and right open, i.e. [left, right).

3. While-loop for intervals, while(left <= right) or while(left < right).

4. Then determine how to move the middle, for left <= right, right = middle - 1, for left < right, right = middle, because right open and no meaning.

```python
def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while(left <= right):
            mid = left + (right - left) // 2
            if (nums[mid] < target):
                left = mid + 1
            elif (nums[mid] > target):
                right = mid - 1
            else:
                return mid
        return -1
```

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] < target)
                left = mid + 1;
            else if(nums[mid] > target)
                right = mid - 1;
            else
                return mid;
        }
        return -1;
    }
};
```





#### 27. Remove Element

Given an integer array `nums` and an integer `val`, remove all occurrences of `val` in `nums` [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm). The order of the elements may be changed. Then return *the number of elements in* `nums` *which are not equal to* `val`.

Consider the number of elements in `nums` which are not equal to `val` be `k`, to get accepted, you need to do the following things:

- Change the array `nums` such that the first `k` elements of `nums` contain the elements which are not equal to `val`. The remaining elements of `nums` are not important as well as the size of `nums`.
- Return `k`.

https://leetcode.cn/problems/remove-element/



- **Double pointer method**(fast and slow pointer method)

Use a fast pointer and a slow pointer to complete the work of two for-loops in one for-loop. 

Fast pointer: Find the elements of the new array, the new array is the array without the target element.
Slow pointer: Point to the position of the updated new array index

```python
def removeElement(self, nums: List[int], val: int) -> int:
    fast = 0  # 快指针
    slow = 0  # 慢指针
    size = len(nums)
    while fast < size:  # 不加等于是因为，a = size 时，nums[a] 会越界
  # slow 用来收集不等于 val 的值，如果 fast 对应值不等于 val，则把它与 slow 替换
         if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
         fast += 1
     return slow
 
def removeElement(self, nums: List[int], val: int) -> int:
        num_diff = 0		# slow pointer
        for i in range(len(nums)):  # fast pointer
            if (nums[i] != val):
                nums[num_diff] = nums[i]
                num_diff += 1
        return num_diff
```

```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slow = 0;
        for(int fast = 0; fast < nums.size(); ++fast){
            if(nums[fast] != val){
                nums[slow] = nums[fast];
                slow++;
            }              
        }
    return slow;
    }
};
```





#### 977. Square of sorted array

Given an integer array `nums` sorted in **non-decreasing** order, return *an array of **the squares of each number** sorted in non-decreasing order*.

https://leetcode.cn/problems/squares-of-a-sorted-array/



```python
def sortedSquares(self, nums: List[int]) -> List[int]:
    return [x * x for x in sorted(nums, key = lambda x: abs(x))]
    return sorted(x * x for x in nums)
```

Notes: 

The `key` parameter in the `sorted` function is used to specify a function that will be called on each element before making comparisons. 

A `lambda` function in Python is to define a small anonymous function using the `lambda`keyword.



```c++
// double pointers -- O(log n)
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        vector<int> result(nums.size(), 0);
        for(int i = 0, j = nums.size() - 1; i <= j; ){
            if( nums[i]*nums[i] <= nums[j]*nums[j]){
                result[j - i] = nums[j]*nums[j];
                --j;              
            }
            else{
                result[j - i] = nums[i]*nums[i];
                ++i;
            }
        }
        return result;
    }
};

// O(n log n)
class Solution {
public:
    vector<int> sortedSquares(vector<int>& A) {
        for (int i = 0; i < A.size(); i++) {
            A[i] *= A[i];
        }
        sort(A.begin(), A.end()); // sorted value: std::sort
        return A;
    }
};
```





#### 209. Subarray with the smallest length

Given an array of positive integers `nums` and a positive integer `target`, return *the **minimal length** of a* *subarray* *whose sum is greater than or equal to* `target`. If there is no such subarray, return `0` instead.

https://leetcode.cn/problems/minimum-size-subarray-sum/description/



- **Sliding Window method**

​      Continuously adjust the starting and ending positions of the subsequence to get the results we want.

Question:

1. <u>What is in the window?</u>
2. <u>How to move the starting position of the window?</u>
3. <u>How to move the ending position of the window?</u>

<img src="https://camo.githubusercontent.com/21698e6ff850dbe505c025b89fede4a2ca4f872364494ee908b3eba479d41b1a/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f676966732f3230392e2545392539352542462545352542412541362545362539432538302545352542302538462545372539412538342545352541442539302545362539352542302545372542422538342e676966" alt="209.长度最小的子数组" style="zoom:67%;" />

*A window is a continuous subarray with the smallest length that satisfies its sum ≥ s.*

*How to move the starting position of the window?   If the current window value is greater than or equal to s, the window will move forward (that is, it should be reduced).*

*How to move the ending position of the window?   The ending position of the window is the pointer to the traversal array, that is, the index in the for loop.*

```c++
class Solution {    // O(n)
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int result = INT32_MAX;
        int index_slide = 0;
        int sum = 0;
        int sub_length = 0;
        for(int i = 0; i < nums.size(); ++i){
            sum += nums[i];
            while(sum >= target){
                sub_length = i - index_slide + 1;
                result = result < sub_length? result : sub_length;
                sum -= nums[index_slide];
                index_slide++;
            }
        }
        return result == INT32_MAX? 0 : result;
    }
};
```

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l_sum = 0
        left = 0
        min_len = float('inf')
        sub_length = 0
        for right in range(len(nums)):
            l_sum += nums[right]
            while l_sum >= target:
                sub_length = right - left + 1
                min_len = min(min_len, sub_length)
                l_sum -= nums[left]
                left += 1
        return min_len if min_len != float('inf') else 0
```

Notes: We set the initial value of result to int32Max and float('inf'), only when sub_length has value will change the value of it. So at the end, we must convert it to return 0.





#### 59. Spiral Matrix II

Given a positive integer `n`, generate an `n x n` `matrix` filled with elements from `1` to `n2` in spiral order.

https://leetcode.cn/problems/spiral-matrix-ii/



- Solution

​	Loop times is a binary search. If you want to write a correct binary search, you must adhere to the principle of loop invariants.

​	Simulate the process of drawing a matrix clockwise:

​		Fill the upper row from left to right
​		Fill the right column from top to bottom
​		Fill the lower row from right to left
​		Fill the left column from bottom to top
​		Draw from outside to inside in circles.

​	We have to draw every four sides. How to draw these four sides?  <u>Each side must adhere to the consistent principle of left closed and right open, or left open and right closed, so that this circle can be drawn according to a unified rule.</u>

<img src="https://camo.githubusercontent.com/fea919fc77fdb6d517e25ae404a9d45a39cc8351e11664ff8112b37bbbc00207/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303232303932323130323233362e706e67" alt="img" style="zoom: 25%;" />

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        nums = [[0] * n for _ in range(n)]
        startx, starty = 0, 0             # 起始点
        loop, mid = n // 2, n // 2        # 迭代次数、n为奇数时，矩阵的中心点
        count = 1                           # 计数

        for offset in range(1, loop + 1) :      
          # 每循环一层偏移量加1，偏移量从1开始
            for i in range(starty, n - offset) :    # 从左至右，左闭右开
                nums[startx][i] = count
                count += 1
            for i in range(startx, n - offset) :    # 从上至下
                nums[i][n - offset] = count
                count += 1
            for i in range(n - offset, starty, -1) : # 从右至左
                nums[n - offset][i] = count
                count += 1
            for i in range(n - offset, startx, -1) : # 从下至上
                nums[i][starty] = count
                count += 1                
            startx += 1         # 更新起始点
            starty += 1

        if n % 2 != 0 :			# n为奇数时，填充中心点
            nums[mid][mid] = count 
        return nums
```

```c++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> result(n, vector<int>(n, 0));
        int count = 1;
        int loop = n / 2;
        int mid = n / 2;
        int startx = 0;
        int starty = 0;
        for(int m_loop = 1; m_loop <= loop; m_loop++){
            for(int i = startx; i < n - m_loop ; i++){
                result[starty][i] = count;
                count++;
            }
            for(int i = starty; i < n - m_loop ; i++){
                result[i][n - m_loop] = count;
                count++;
            }
            for(int i = n - m_loop; i > startx ; i--){
                result[n - m_loop][i] = count;
                count++;
            }
            for(int i = n - m_loop; i > starty ; i--){
                result[i][startx] = count;
                count++;
            }
            startx++;
            starty++;
        }
        if(n % 2 != 0){
            result[mid][mid] = count;
        }
        return result;
    }
};
```

Notes: the fixed index like x and y, and n - loop. Also, the mid point.





#### Interval Sum

Given an integer array `Array`, calculate the sum of the elements in each specified interval of the array.

Input description: The first line of input is the length `n` of the integer array `Array`, followed by n lines, each with an integer, representing the elements of the array. Subsequent input is the interval for which the sum needs to be calculated, until the end of the file.

Output description: Output the sum of the elements in each specified interval.

https://kamacoder.com/problempage.php?pid=1070



- **Prefix Sum** : useful in calculating interval sums

The idea of prefix sum is to <u>reuse the sum of the calculated subarrays, thereby reducing the number of cumulative calculations required for interval queries.</u>

Prefix sum is very useful when it comes to calculating interval sums!

```c++
#include <iostream>
#include <vector>

int main()
{
    int n, a, b;
    std::cin >> n;
    std::vector<int> arr(n);
    std::vector<int> presum(n);
    int sum = 0;
    for( int i = 0; i < n ; ++i){
        std::cin >> arr[i];
        sum += arr[i];
        presum[i] = sum;
    }
    while( std::cin >> a >> b){
        if( a == 0) 
            std::cout << presum[b] << std::endl;
        else 
            std::cout << presum[b] - presum[a - 1] << std::endl;
    }
    return 0;
}
```

```c++
#include <iostream>
#include <vector>

int main()
{
    int n, a, b;
    std::cin >> n;
    std::vector<int> arr(n, 0) ;
    for( int i = 0; i < n ; ++i){
        std::cin >> arr[i];
    }
    while( std::cin >> a >> b){
        int sum = 0;
        for(int i = a; i <= b; i++)
            sum += arr[i];
        std::cout << sum << std::endl;
    }
    
    return 0;
}   // O(n * m), not recommend
```

```python
def main():
    import sys
    input = sys.stdin.read()
    data = input.split()
    
    idx = 0
    n = int(data[idx])
    idx += 1
    
    sum = 0
    presum = []
    for i in range(n):
        sum += int(data[idx])
        presum.append(sum)
        idx += 1
    
    result = []
    while idx < len(data):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        
        if a == 0:
            print( presum[b])
        else:
            print( presum[b] - presum[a - 1])
            
if __name__ == '__main__':
    main()
```





#### Developers purchase land

​	In a city area, it is divided into **n * m** continuous blocks, each block has a different weight, representing its land value. Currently, there are two development companies, Company A and Company B, who want to purchase land in this city area.

​	Now, all the blocks in this city area need to be allocated to Company A and Company B. However, due to the restrictions of urban planning, the area is only allowed to be divided into two sub-areas horizontally or vertically, and each sub-area must contain one or more blocks.

​	To ensure fair competition, you need to find a way to allocate so that the difference in the total value of land in the respective sub-areas of Company A and Company B is minimized.

​	Note: Blocks cannot be divided any further.

[Input description]

​	Input two positive integers in the first line, representing n and m.

​	In the next n lines, each line outputs m positive integers.

[Output description]

​	Please output an integer representing the minimum difference between the total value of land in the two sub-areas.

https://kamacoder.com/problempage.php?pid=1044



Solution:

​	Use the idea of prefix sum to solve it. First, find the sum in the row direction and the sum in the column direction. This way, you can easily know the sum of the two intervals divided by .

```c++
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

int main(){
    int n, m;
    cin >> n >> m;
    vector<vector<int>> area(n, vector<int> (m, 0));
    int sum = 0;
    for(int i = 0; i < n; i++){
        for (int j = 0; j < m; j++) {
            cin >> area[i][j];
            sum += area[i][j];
        }
    }			// O(n * m)
    
    vector<int> horizontal(n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0 ; j < m; j++) {
            horizontal[i] += area[i][j];
     // 遍历到行末尾时候开始统计
     // if (j == m - 1) result = min (result, abs(sum - count - count));
        }
    }			// O(n * m)
    
    vector<int> vertical(m , 0);
    for (int j = 0; j < m; j++) {
        for (int i = 0 ; i < n; i++) {
            vertical[j] += area[i][j];
     // 遍历到列末尾的时候开始统计
     // if (i == n - 1) result = min (result, abs(sum - count - count));
        }
    }			// O(n * m)
    
    int result = INT_MAX;
    int horizontalCut = 0;
    for (int i = 0; i < n; i++) {
        horizontalCut += horizontal[i];
        result = min(result, abs(sum - horizontalCut - horizontalCut));
    }			// O(n)
    
    int verticalCut = 0;
    for (int j = 0; j < m; j++) {
        verticalCut += vertical[j];
        result = min(result, abs(sum - verticalCut - verticalCut));
    }			// O(m)
  
    std::cout << result << std::endl;
    
    return 0;
}			// 3 O(n * m) + O(n) + O(m) ~= O(n * m)
```

```python
def main():
    import sys
    data = sys.stdin.read().split()
    
    idx = 0
    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx += 1
    sum = 0
    vec = []
    
    # sum of area
    for i in range(n):
        row = []
        for j in range(m):
            num = int(data[idx])
            row.append(num)
            sum += num
            idx += 1
        vec.append(row)
    
    result = float('inf')
    
    # horizontal
    row_sum = 0
    for i in range(n):
        for j in range(m):
            row_sum += vec[i][j]
            if( j == m - 1):
                result = min(result, abs(sum - 2 * row_sum))
            
    # vertical
    col_sum = 0
    for j in range(m):
        for i in range(n):
            col_sum += vec[i][j]
            if( i == n - 1):
                result = min(result, abs(sum - 2 * col_sum))       
    
    print(result)    

if __name__ == "__main__":
    main()    
```



## 2. Linked List

#### Summary

![img](https://code-thinking-1253855093.file.myqcloud.com/pics/%E9%93%BE%E8%A1%A8%E6%80%BB%E7%BB%93.png)

**A linked list is a linear structure connected together by pointers.** 

Each node is composed of <u>two parts, one is a data, one is a pointer (storing pointers to the next node)</u>. The pointer of the last node points to null pointer (empty pointer).

The entry node of the linked list is called the **head** node of the linked list.

![链表1](https://camo.githubusercontent.com/829520739bac1ba45419058035cb0c6174bf996c98b8f15d795741f540597125/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303830363139343532393831352e706e67)

##### Type of Linked List

1. Singly-linked list

​	The pointer of every node only points to the next node.

2. Double linked list

​	Has two pointers which point to the previous and the next node.

![链表2](https://camo.githubusercontent.com/5fa11a1ed756d7041d5566b05863b2e9db1bb97476823ae5df32a714713af1ab/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303830363139343535393331372e706e67)

3. Circular linked list

​	The head and the tail of the linked list connect.

<img src="https://camo.githubusercontent.com/2b41ca195ae0a57b2e0b9fee108b0b418e28e7fa0be9ca4525047defbdc648d6/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303830363139343632393630332e706e67" alt="链表4" style="zoom: 67%;" />



##### Storage Type of Linked List

​	Linked list is not stored in contiguous memory space, but connected by pointers to every node, so stored in random memory space which depends on storage management of system.

![链表3](https://camo.githubusercontent.com/4ffa918cfcb0b99cdde87cfa6479c953b4bbc3bb6e6ecc16cd8b0055838b9001/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303830363139343631333932302e706e67)



##### **Definition in C++ and python**

```c++
struct ListNode {
    int val;  // value stored in node
    ListNode *next;  // pointer to next node
    ListNode(int x) : val(x), next(NULL) {}  // constructor
};
```

If we don't define the constructor, we cannot initilize the value easily.

```c++
ListNode* head = new ListNode(5);

// if use default constructor
ListNode* head = new ListNode();
head->val = 5;
```



```python
class ListNode:
    def __init__(self, val = 0, next=None):
        self.val = val
        self.next = next
```



##### Addition and Deletion : O (1)

1. Addition

![链表-添加节点](https://camo.githubusercontent.com/b8b7beb6e90438ff64d73dcbc815dc14a1c9ed7176b4fe2116dfa3aeed74b578/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303830363139353133343333312d32303233303331303132313530333134372e706e67)

​	change the pointer to the new node, and modify the pointer of the new node to the next node.

2. Deletion

![链表-删除节点](https://camo.githubusercontent.com/b50b95dce3c3b4a75690369b0156e97d4e3b88b4f3a93fcac1cfb6304f24fd64/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303830363139353131343534312d32303233303331303132313435393235372e706e67)

​	change the pointer to the next node, and release the memory of deleted node.

Notes: 

​	<u>If we delete the last node, we need to iterate from the head, and find the node before the last node, then delete the pointer of this node and this node become the last node. ( O (n) )</u>



##### Comparison of arrays and linked list

<img src="https://camo.githubusercontent.com/90cecf4ba7567d6374ee9a4b14db047c47ea6fa206e59a72254c42cf4a597f76/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303830363139353230303237362e706e67" alt="链表-链表与数据性能对比" style="zoom: 50%;" />

**The length of an array is fixed when defined.** If you want to change the length of the array, you need to redefine a new array.

**The length of the linked list can be unfixed, and can be dynamically added or deleted,** suitable for the scenario where the amount of data is not fixed, frequent addition or deletion, and few queries.





#### Main methods

##### ==dummy head==

<u>If we want to deal with the current node, we must find the previous node. But for the head, it doesn't have the previous node, so we need to define a dummy head.</u>

Otherwise, we need to **<u>conside the head is NULL and the head is the key node we will solve</u>**, and we can refer to the examples.



##### ==Iteration==



##### ==Recursion==





#### 203. Remove Linked List elements

Given the `head` of a linked list and an integer `val`, remove all the nodes of the linked list that has `Node.val == val`, and return *the new head*.

https://leetcode.cn/problems/remove-linked-list-elements/



- **==dummy head node==** 

Time complexity:  O( n ).           Space complexity:  O( 1 )

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy_head = ListNode(next=head)
        cur = dummy_head
        while cur.next != None :	   ## Important!!!
            if cur.next.val == val :
                cur.next = cur.next.next
            else:
                cur = cur.next
        
        return dummy_head.next
```

Notes: 

​	If the head is the node we need to move, then we must let the next of the previous node equal to the next node. So we use dummy head node.



- **==Recursion==** ( direct ways )

We have a loop and repeat it. Do recursion don't need the dummy head.

==For recursion, <u>the first point is to consider the first condition==(test the head node)</u> , ==the second point is to <u>writer the command we need and do recursion in this condition==(test the next node and so on)</u>.

Time complexity:  O(n)        Space complexity:  O(n)

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if head == None:
            return head
        if head.val == val:
            newHead = self.removeElements(head.next, val)
            return newHead
        else:
            head.next = self.removeElements(head.next, val)
            return head
```

Notes:

​	**When we don't use dummy head, we must consider the head node is NULL and the head node is what we want to remove.** 

​	why use head->next? 

​	Because the head now is actually the head we need to return, but we need to <u>check the remaining part by recursion. Just let the next node to be the "head".</u>



- in C++

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode* dummy_head = new ListNode(0, head);

        ListNode* cur_node = dummy_head;
        while( cur_node -> next != nullptr){
            if (cur_node -> next -> val == val){
                ListNode* tmp = cur_node -> next;
                cur_node -> next = tmp -> next;
                delete tmp;
            }
            else{
                cur_node = cur_node -> next;
            }
        }
        head = dummy_head -> next;
        delete dummy_head;
        return head;   
    }
};
// Time complexity: O(n)    Space complexity: O(1)
```

Notes: Remember to release the memory space !!!

```c++
// Recursion
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if (head == nullptr) {
            return nullptr;
        }		// also consider the last node points to empty pointer.

        if (head->val == val) {
            ListNode* newHead = removeElements(head->next, val);
            delete head;
            return newHead;
        } else {
            head->next = removeElements(head->next, val);
            return head;
        } 
    }
};
// Time complexity: O(n)   Space complexity: O(n)
```

```c++
// delete directly
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        // 删除头结点
        while (head != NULL && head->val == val) { // 注意这里不是if
            ListNode* tmp = head;
            head = head->next;
            delete tmp;
        }

        // 删除非头结点
        ListNode* cur = head;
        while (cur != NULL && cur->next!= NULL) {
            if (cur->next->val == val) {
                ListNode* tmp = cur->next;
                cur->next = cur->next->next;
                delete tmp;
            } else {
                cur = cur->next;
            }
        }
        return head;
    }
};
// Time complexity: O(n)    Space complexity: O(1)
```





#### 707. Design Linked List

Design your implementation of the linked list. You can choose to use a singly or doubly linked list.
A node in a singly linked list should have two attributes: `val` and `next`. `val` is the value of the current node, and `next` is a pointer/reference to the next node.
If you want to use the doubly linked list, you will need one more attribute `prev` to indicate the previous node in the linked list. Assume all nodes in the linked list are **0-indexed**.

Implement the `MyLinkedList` class:

- `MyLinkedList()` Initializes the `MyLinkedList` object.
- `int get(int index)` Get the value of the `indexth` node in the linked list. If the index is invalid, return `-1`.
- `void addAtHead(int val)` Add a node of value `val` before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
- `void addAtTail(int val)` Append a node of value `val` as the last element of the linked list.
- `void addAtIndex(int index, int val)` Add a node of value `val` before the `indexth` node in the linked list. If `index` equals the length of the linked list, the node will be appended to the end of the linked list. If `index` is greater than the length, the node **will not be inserted**.
- `void deleteAtIndex(int index)` Delete the `indexth` node in the linked list, if the index is valid.

https://leetcode.cn/problems/design-linked-list/description/



- Two ways to operate a linked list:

Use the original linked list directly to operate.
**<u>Set a virtual header node to perform operations.( Usually easy !!!)</u>**



- **For Singly-Linked List**

Time Complexity:  O( index )

Space complexity:  O(n)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
      
class MyLinkedList:

    def __init__(self):
        self.head = ListNode()
        self.size = 0   

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1

        curNode = self.head
        for i in range(index + 1):
            curNode = curNode.next
        
        return curNode.val

    def addAtHead(self, val: int) -> None:
      	newNode = ListNode(val, self.head.next)
        self.head.next = newNode
        self.size += 1       

    def addAtTail(self, val: int) -> None:
        curNode = self.head
        while curNode.next:
            curNode = curNode.next
        curNode.next = ListNode(val)
        self.size += 1
        
    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return -1
        
        curNode = self.head
        for i in range(index):
            curNode = curNode.next
        curNode.next = ListNode(val, curNode.next)
        self.size += 1
        
    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return -1
        
        curNode = self.head
        for i in range(index):
            curNode = curNode.next
        curNode.next = curNode.next.next
        self.size -= 1
```

Notes:

1. **for linked list, we need to know the head. So we must define the head and initilize it**. Besides, for index, we need to define size and initilize.
2. for linked list, **the loop condition usually is  while( curNode.next )**.



- **Doubly Linked List**

<u>Don't need dummy one, we can just use the exact one.</u>

```python
class MyLinkedList:

    def __init__(self):
        self.head = ListNode()	# exact head, not dummy one 
        self.tail = ListNode()	# exact tail, not dummy one 
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1

        if index < self.size // 2:		# mid one
            curNode = self.head
            for i in range(index):
                curNode = curNode.next
        else:
            curNode = self.tail
            for i in range(self.size - 1 - index):
                curNode = curNode.prev
        return curNode.val

    def addAtHead(self, val: int) -> None:
        newNode = ListNode(val, None, self.head)
        if self.head:
            self.head.prev = newNode
        else:
            self.tail = newNode
        self.head = newNode
        self.size += 1

    def addAtTail(self, val: int) -> None:
        newNode = ListNode(val, self.tail, None)
        if self.tail:
            self.tail.next = newNode
        else:
            self.head = newNode
        self.tail = newNode
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return 
        
        if index == 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        else:
            if index < self.size // 2:
            		curNode = self.head
            		for i in range(index - 1):
                		curNode = curNode.next
        		else:
            		curNode = self.tail
            		for i in range(self.size - 1 - (index - 1)):
                		curNode = curNode.prev
        		newNode = ListNode(val, curNode, curNode.next)
        		curNode.next.prev = newNode
        		curNode.next = newNode
        		self.size += 1
            
    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return 
        
        if index == 0:	# ensure if head is empty node
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None
        elif index == self.size - 1:	# ensure if tail is empty node 
            self.tail = self.tail.prev
            if self.tail:
                self.tail.next = None
            else:
                self.head = None
        else:
            if index < self.size // 2:
                current = self.head
                for i in range(index):
                    current = current.next
            else:
                current = self.tail
                for i in range(self.size - index - 1):
                    current = current.prev
            current.prev.next = current.next
            current.next.prev = current.prev
        self.size -= 1
```

Note:

**Why we need *index == 0* and *index == self.size - 1* condition in deleteAtIndex?**

If current.prev is None, eg. current is the head, then we cannot do current.prev.next

If current.next is None, eg. current is the tail, then we cannot do current.next.prev

Note:

**Why we need \*index == 0\* and \*index == self.size - 1\* condition in deleteAtIndex?**

If current.prev is None, eg. current is the head, then we cannot do current.prev.next

If current.next is None, eg. current is the tail, then we cannot do current.next.prev



- **For Singly-Linked List in C++**

```c++
class MyLinkedList {
public:
    /**
  	struct LinkedNode {
        int val;
        LinkedNode* next;
        LinkedNode(int val):val(val), next(nullptr){}
    };  **/

    MyLinkedList() {
        m_Head = new ListNode(0);
        m_size = 0;
    }
    
    int get(int index) {
        if(index < 0 || index >= m_size){ return -1; }

        ListNode* curNode = m_Head;
        for(int i = 0;i <= index; i++){
            curNode = curNode->next;
        }
        return curNode->val;
    }
    
    void addAtHead(int val) {
        ListNode* newNode = new ListNode(val);
        newNode->next = m_Head->next;
        m_Head->next = newNode;
        m_size++;
    }
    
    void addAtTail(int val) {
        ListNode* curNode = m_Head;
        while(curNode->next){
            curNode = curNode->next;
        }
        ListNode* newNode = new ListNode(val);
        newNode->next = curNode->next;
        curNode->next = newNode;
        m_size++;
    }
    
    void addAtIndex(int index, int val) {
        if(index < 0 || index > m_size){ return ; }

        ListNode* curNode = m_Head;
        for(int i = 0;i < index; i++){
            curNode = curNode->next;
        }
        ListNode* newNode = new ListNode(val);
        newNode->next = curNode->next;
        curNode->next = newNode;
        m_size++;
    }
    
    void deleteAtIndex(int index) {
        if(index < 0 || index >= m_size){ return ; }

        ListNode* curNode = m_Head;
        for(int i = 0;i < index; i++){
            curNode = curNode->next;
        }
        ListNode* tmp = curNode->next;
        curNode->next = curNode->next->next;
        delete tmp;
        m_size--;
    }

private:
    ListNode* m_Head;
    int m_size;
};
```



### [206. Reverse Linked List](https://leetcode.cn/problems/reverse-linked-list/description/)

Given the `head` of a singly linked list, reverse the list, and return *the reversed list*.



#### Double Pointers method 

​	You just need to ==**change the next pointer of the linked list**==, and directly reverse the linked list without redefining a new linked list (consume more memory space).

​	We need to ==use the current node and the previous node,and renew them== . These are two pointers.



#### Reversion

​	<u>We use loop and continue to renew the curNode and preNode, so we consider the recursion</u>. Every reverse is a recursion.

​	We revise the operation within the loop to a seperate function.

​	The inputs are the variables its operation needs.

​	Then return the next recursion. eg. return self.FuncName( x1, x2)

​	The loop exit condition will be put in the beginning of this function, and return the final result.



#### Solution

- ==O( n ) / O( 1 )==

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curNode = head
        PreNode = None
        while curNode:           
            tmp = curNode.next	# store the next node
            curNode.next = PreNode
            PreNode = curNode
            curNode = tmp
        return PreNode
```

Notes:  **<u>Why save curNode.next ?</u>** 

​	because we're going to change the pointing of the cur->next.



- ==O( n ) / O( n )==

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        return self.reverse(head, None)
    
    def reverse(self, cur: ListNode, pre: ListNode) -> ListNode:
        if cur == None:
            return pre

        tmp = cur.next
        cur.next = pre
        return self.reverse(tmp, cur)
```



- Double Pointers in C++

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* curNode = head;
        ListNode* preNode = nullptr;
        while(curNode){
            ListNode* tmp = curNode->next;
            curNode->next = preNode;
            preNode = curNode;
            curNode = tmp;
        }
        return preNode;
    }
};
```





#### 24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

https://leetcode.cn/problems/swap-nodes-in-pairs/



- **Solution:**

==**We must draw a picture to simulate the process of swap and every step.**==

Follow the left to the right side.

![24.两两交换链表中的节点1](https://camo.githubusercontent.com/5d91c44c49f123047a51a63b2d36c271e225ec7f0110ed623c93bcf94d17ee98/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f706963732f32342e254534254238254134254534254238254134254534254241254134254536253844254132254539253933254245254538254131254138254534254238254144254537253941253834254538253841253832254537253832254239312e706e67)



- **Dummy Head Node**

==<u>For the head node, we don't have a previous pointer to it, so we need a specific part to handle it seperately. To simplify it, we set a virtual head node which points to it.</u>==

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(next=head)
        curNode = dummy_head

        while curNode.next and curNode.next.next:
            tmp = curNode.next
            tmp1 = curNode.next.next.next
            
            curNode.next = curNode.next.next
            curNode.next.next = tmp
            curNode.next.next.next = tmp1
            curNode = curNode.next.next
        
        return dummy_head.next
```

Note: we need to change the pointer of node 2 and node 1 to node 1 and node 3, so **==setting tmp node to node 1 and node 3 to ensure it unchanged.==**

 If remaining zero node, while curNode.next != 0. If remaining one node, curNode.next.next != 0. If remaining two node, can do swap.



- **Recursion**

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        
        pre = head
        cur = head.next
        next = cur.next

        cur.next = pre
        pre.next = self.swapPairs(next)
        return cur
```

==**If we use recursion, usually we don't need the dummy head**.==

But, <u>**we need to consider the head is none, or the head.next is none**</u>, which will influence the swap.</u> 

Be careful that the pre.next is the fourth node! Not the third one.



- **in C++**

```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode* dummy_head = new ListNode(0, head);
        
        ListNode* curNode = dummy_head;
        while(curNode->next != nullptr && curNode->next->next != nullptr){
            ListNode* tmp1 = curNode->next;
            ListNode* tmp2 = curNode->next->next->next;

            curNode->next = curNode->next->next;
            curNode->next->next = tmp1;
            curNode->next->next->next = tmp2;
            curNode = curNode->next->next;
        }
        ListNode* newHead = dummy_head->next;
        delete dummy_head;

        return newHead;
    }
};
```





### [19. Remove Nth Node From End of List](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/)

Given the `head` of a linked list, remove the `nth` node from the end of the list and return its head.



#### ==**Fast-Slow / Double Pointers**==

If we want to delete the penultimate Nth node, let ==**fast pointer move n steps**==, then let ==fast and slow pointer move at the same time **until fast points to the end of the list**==. 

To delete, we must <u>let the previous node points to the next node,</u> but **if we only have one node, we cannot specify the previous node, that's why we need dummy head node.**

<u>Under this condition, we can let the slow pointer to previous node(dummy head), and then the fast pointer must move ( n + 1) steps</u>. When the fast pointer reaches the end / null pointer, we delete current node by the previous node.



#### Solution

==O(n) / O(1)==

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy_head = ListNode(next = head)

        fast = dummy_head
        slow = dummy_head
        for i in range(n + 1):
            fast = fast.next
        while fast:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummy_head.next
```

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy_head = new ListNode(0, head);
        
        ListNode* fast = head;
        for(int i = 1; i <= n; i++){
            fast = fast->next;
        }
        ListNode* slow = dummy_head;
        while(fast != NULL){
            fast = fast->next;
            slow = slow->next;
        }
        slow->next = slow->next->next;
        ListNode* newHead = dummy_head->next;
        delete dummy_head;
        return newHead;
    }
};
```





### [160. Intersection of Two Linked Lists](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/)

Given the heads of two singly linked-lists `headA` and `headB`, return *the node at which the two lists intersect*. If the two linked lists have no intersection at all, return `null`.



#### Idea

The intersection part means ==**the  beginning of the pointers in the intersection part are the same in both A and B**==

1. We <u>calculate the length of two linked list, and the difference (b - a) of their length. The move curA (b - a) steps and now their tail are aligned.</u>

<img src="https://code-thinking.cdn.bcebos.com/pics/%E9%9D%A2%E8%AF%95%E9%A2%9802.07.%E9%93%BE%E8%A1%A8%E7%9B%B8%E4%BA%A4_2.png" alt="面试题02.07.链表相交_2" style="zoom:65%;" />

2. <u>Compare curA and curB. If not equal, move two until they are equal, the point is the intersection.</u> Otherwise return the null pointer.



#### Solution

==O(n + m) / o(1)==

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        curNode = headA
        size_A = 0
        while curNode:
            size_A += 1
            curNode = curNode.next
        
        curNode = headB
        size_B = 0
        while curNode:
            size_B += 1
            curNode = curNode.next
        
        curA, curB = headA, headB
        if size_A > size_B:
            for i in range(size_A - size_B):
                curA = curA.next
        else:
            for i in range(size_B - size_A):
                curB = curB.next
                
        while curA:
            if curA == curB:
                return curA
            curA = curA.next
            curB = curB.next

        return None
```



- in C++

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* curNode = headA;
        int size_A = 0;
        while(curNode != nullptr){
            curNode = curNode->next;
            size_A++;
        }

        curNode = headB;
        int size_B = 0;
        while(curNode != nullptr){
            curNode = curNode->next;
            size_B++;
        }

        ListNode* curA = headA;
        ListNode* curB = headB;
        if(size_A - size_B > 0){
            for(int i = 0; i < size_A - size_B; i++){
                curA = curA->next;
            }
        }
        else{
            for(int i = 0; i < size_B - size_A; i++){
                curB = curB->next;
            }
        }

        while(curA != nullptr){
            if(curA == curB){
                return curA;
            }
            curA = curA->next;
            curB = curB->next;
        }
        return nullptr;
    }
};
```





### 142. Linked List Cycle II

Given the `head` of a linked list, return *the node where the cycle begins. If there is no cycle, return* `null`.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to (**0-indexed**). It is `-1` if there is no cycle. **Note that** `pos` **is not passed as a parameter**.

**Do not modify** the linked list.

https://leetcode.cn/problems/linked-list-cycle-ii/description/



- **Solution**

1. How to determine whether it has a cycle ?

**==Fast-Slow pointers.==**

Start from the head node, <u>every time the fast pointer moves 2 steps, the slow pointer moves 1 step. If they meet, it has an cycle.</u>

Why? **If they meet, it must in the cycle, because the fast moves 1 node relative to the slow.** We can draw a cycle, and let the fast starts chasing the slow, we will find all the situation is as following:

<img src="https://code-thinking-1253855093.file.myqcloud.com/pics/20210318162236720.png" alt="142环形链表1" style="zoom:67%;" />

<img src="https://code-thinking.cdn.bcebos.com/gifs/141.%E7%8E%AF%E5%BD%A2%E9%93%BE%E8%A1%A8.gif" alt="141.环形链表" style="zoom:90%;" />

2. How to find the entry of the cycle?

We assume the number of node between the head and entry is X , the number of node between the entry and node they meet is Y , the number of node between the node they meet and the entry again is Z.

<img src="https://code-thinking-1253855093.file.myqcloud.com/pics/20220925103433.png" alt="img" style="zoom:67%;" />

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20250102163032448.png" alt="image-20250102163032448" style="zoom: 50%;" />

<img src="https://code-thinking.cdn.bcebos.com/gifs/142.%E7%8E%AF%E5%BD%A2%E9%93%BE%E8%A1%A8II%EF%BC%88%E6%B1%82%E5%85%A5%E5%8F%A3%EF%BC%89.gif" alt="142.环形链表II（求入口）" style="zoom:75%;" />

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20250102163253360.png" alt="image-20250102163253360" style="zoom:50%;" />

**x = (n - 1) (y + z) + z.** 注意这个<u>x是slow指针走到入口的距离。如果相遇节点有一个指针，走z步也可以到这个入口，此时这两个指针相遇了，就找到入口了</u>。如果cycle了很多圈，就是相遇节点的这个指针在环里多走了n-1圈。

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

            if fast == slow:
                index1 = fast
                index2 = head

                while index1 != index2:
                    index1 = index1.next
                    index2 = index2.next
                
                return index1
      
        return None
```



- Simple solution in python 

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        visited  = set()

        while head:
            if head in visited:
                return head
            visited.add(head)
            head = head.next

        return None
```

Python is a good tool to do data analysis! Store and find it easily!



- in C++

```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* fast = head;
        ListNode* slow = head;

        while(fast != nullptr && fast->next != nullptr){
            fast = fast->next->next;
            slow = slow->next;

            if(fast == slow){
                ListNode* index1 = fast;
                ListNode* index2 = head;
                
                while(index1 != index2){
                    index1 = index1->next;
                    index2 = index2->next;
                }
                return index1;
            }
        }
        return NULL;
    }
};
```





## 3. Hash Table

#### Basic knowledge

1. ==A hash table is a data structure that **directly access the value based on the key**.==

eg. the array is a hash table and the key is index.



2. Frequent usage: ==**<u>Determine whether an element exists in the collections.</u>**==

**==Preference : unordered_set==**, because the time for access, addition, deletion is optimal.

If **need ordered collection**, use **set**.

If **need ordered collection and duplicates**, use **multiset**.



3. ==**Hash function: map some keys to the index of hash table by hash code**==

For keys to be the index, convert it to a number by hash code. Hash code is a specific encoding method that can convert other data formats.

<u>If the hash code is bigger than hash table size, take modulus to map it.</u>

<img src="https://code-thinking-1253855093.file.myqcloud.com/pics/2021010423484818.png" alt="哈希表2" style="zoom: 60%;" />



4. ==**Hash Collision**: Two keys are mapped to the same index==

**Solution 1: Zipper Method**

<u>All the collided elements will be stored in same index by a linked list</u>, so we can search the index and then search the linked list. 

Suitable table size!!! Less empty value of array and short linked list.

<img src="https://code-thinking-1253855093.file.myqcloud.com/pics/20210104235015226.png" alt="哈希表4" style="zoom:55%;" />

**Solution 2: Linear Detection Method**

<u>Must: Table size > Data size</u>

<u>If two indexs collide, stored another one in the next position.</u>

<img src="https://code-thinking-1253855093.file.myqcloud.com/pics/20210104235109950.png" alt="哈希表5" style="zoom:55%;" />



5. Three common hash structures

(1) **Arrays**               

<img src="https://camo.githubusercontent.com/90cecf4ba7567d6374ee9a4b14db047c47ea6fa206e59a72254c42cf4a597f76/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303230303830363139353230303237362e706e67" alt="链表-链表与数据性能对比" style="zoom: 45%;" />

 (2) **Sets** : value is the key.

![image-20250121105443592](/Users/annahuang/Library/Application Support/typora-user-images/image-20250121105443592.png)

(3) **Map** : use key to access value.

![image-20250121105549321](/Users/annahuang/Library/Application Support/typora-user-images/image-20250121105549321.png)





#### 242. Valid Anagram

Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise.

https://leetcode.cn/problems/valid-anagram/description/



- **Solution:**

==**Anagram**: the number of every letter is equal in two strings, but order different==

So we count the number of letters in two strings and compare them. ==**<u>Since the letter is from a to z, we map it to an array, and contiguous index decribes the order( a -> 0, b -> 1...)</u>**==. This is the hash code we specify.

We don’t need to remember the ASCII of character a, we just need to find a relative value( related to order of "a").

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        record = [0] * 26
        for i in s:
            record[ord(i) - ord('a')] += 1
        for i in t:
            record[ord(i) - ord('a')] -= 1
        for j in record:
            if j != 0:
                return False
        return True
```

**Time complexity: O(n + n + 26) = O(2n + 26) = O(n)**

**space complexity: O(1)**



Easiest Way:

```python
if sorted(s) != sorted(t):
     return False
return True
```

In C++:

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        int record[26] = {0};
        for (int i = 0; i < s.size(); i++) {
            // 并不需要记住字符a的ASCII，只要求出一个相对数值就可以了
            record[s[i] - 'a']++;
        }
        for (int i = 0; i < t.size(); i++) {
            record[t[i] - 'a']--;
        }
        for (int i = 0; i < 26; i++) {
            if (record[i] != 0) {
                return false;
            }
        }
        return true;
    }
};
```





#### 1002. Find Common characters

Given a string array `words`, return *an array of all characters that show up in all strings within the* `words` *(including duplicates)*. You may return the answer in **any order**.

https://leetcode.cn/problems/find-common-characters/description/



- Solution

**We need to count the frequency of characters in every word. And choose the common character with minimum frequency.**

==Count characters, use hash table with size 26 to represent a to z.== 

To avoid comparison to get common character, we can store in an array with size 26.

To get the minimum, we can set a dynamic array to compare for every word.

<u>Note!!! The initial frequency is 0 for every 26 character, use the first one to initilize.</u>

<img src="https://camo.githubusercontent.com/07df96e7fdb4558e163884ccbc552af03232869f47c6c7c9a4f7df4c52b73f9c/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f706963732f313030322e2545362539462541352545362538392542452545352542382542382545372539342541382545352541442539372545372541432541362e706e67" alt="1002.查找常用字符" style="zoom: 50%;" />

```python
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        min_hash = [0] * 26

        for char in words[0]:
            min_hash[ord(char) - ord('a')] += 1
        
        for i in range(1, len(words)):
            char_hash = [0] * 26
            for char in words[i]:
                char_hash[ord(char) - ord('a')] += 1
            
            for j in range(26):
                if min_hash[j] > char_hash[j]:
                    min_hash[j] = char_hash[j] 
        
        result = []
        for i in range(26):
            while min_hash[i] != 0:
                result.append(chr(i + ord('a')))
                min_hash[i] -= 1

        return result
```



- Other methods

==import collections==

==collections.Counter(words)  --> output: key + frequency==  Counter({'l': 2, 'e': 1})

==Get intersection :  tmp & collections.Counter(words[i])==

```python
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        tmp = collections.Counter(words[0])
        l = []
        for i in range(1,len(words)):
            # 使用 & 取交集
            tmp = tmp & collections.Counter(words[i])

        # 剩下的就是每个单词都出现的字符（键），个数（值）
        for j in tmp:
            v = tmp[j]
            while(v):
                l.append(j)
                v -= 1
        return l
```



in C++:

```c++
class Solution {
public:
    vector<string> commonChars(vector<string>& words) {
        vector<int> min_hash(26);
        for(int i = 0; i < words[0].size() ; i++){
            min_hash[words[0][i] - 'a']++;
        }

        for(int i = 1; i < words.size() ; i++){
            vector<int> char_hash(26);
            for(int j = 0; j < words[i].size() ; j++){
                char_hash[words[i][j] - 'a']++;
            }
            
            for(int k = 0; k < 26 ; k++){
                if(min_hash[k] > char_hash[k])
                    min_hash[k] = char_hash[k];}    
        }

        vector<string> result;
        for(int i = 0; i < 26 ; i++){
            while (min_hash[i] != 0) { 
                string s(1, i + 'a'); 
                result.push_back(s);
                min_hash[i]--;
            }
        }
        return result;
    }
};
```





#### 349. Intersection of Two Arrays

Given two integer arrays `nums1` and `nums2`, return an array of their intersection. Each element in the result must be **unique** and you may return the result in **any order**.

https://leetcode.cn/problems/intersection-of-two-arrays/description/



- Note element unique and result can be in any order(no duplicates and unordered)

==If we know the range of value==

```python
def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    count_1 = [0] * 1001
    result = []

    for num in nums1:
        count_1[num] = 1
        
    for num in nums2:
        if count_1[num] == 1:
           result.append(num)
           count_1[num] = 0

    return result
  
Time complexity : O(m + n)
Space complexity : O(n)
```

==If we don't know the range of value==

```python
# use set
def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    table = {}
    for num in nums1:
        table[num] = table.get(num, 0) + 1
        
    res = set()
    for num in nums2:
        if num in table:
            res.add(num)
            del table[num]
        
    return list(res)
```

```python
# use existed method in set
# remember to convert set to the output format: list
list(set(nums1) & set(nums2)) 
```



- ==**When use arrays and when use set ?**==

**If the problem limit the size or range of value, we can use arrays.**

But **if the problem doesn't limit the size, and hash value are few** and particularly scattered, and have a very large span, using an array will result in a waste of space.

So we **use set : std::set  /  std::multiset  /  std::unordered_set ( efficient)**



in C++

```c++
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> result_set;
        unordered_set<int> nums_set(nums1.begin(), nums1.end());
        for(int num: nums2){
            if(nums_set.find(num) != nums_set.end()){
                result_set.insert(num);
            }
        }
        return vector<int>(result_set.begin(), result_set.end());
    } 
```





#### 202. Happy Numbers

Write an algorithm to determine if a number `n` is happy.

A **happy number** is a number defined by the following process:

- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it **loops endlessly in a cycle** which does not include 1.
- Those numbers for which this process **ends in 1** are happy.

Return `true` *if* `n` *is a happy number, and* `false` *if not*.

https://leetcode.cn/problems/happy-number/description/



==loops endlessly in a cycle means the sum will appear again.==

To find the sum of the squares of its digits:

1. repeat division by 10, the remainder is the digit. ( 11% 2 )
2. Use divmod(n, 10), return quotient and remainder. (n, r =divmod(n, 10)  )
3. **str(n), every digit become a letter, then use int( ) to convert.**

```python
class Solution:
   def isHappy(self, n: int) -> bool:
       seen = []
       while n != 1:
           n = sum(int(i) ** 2 for i in str(n))
           if n in seen:
               return False
           seen.append(n)
       return True
```

```python
class Solution:
   def isHappy(self, n: int) -> bool:
       seen = set()		# use set
       while n != 1:
           n = sum(int(i) ** 2 for i in str(n))
           if n in seen:
               return False
           seen.add(n)		# for set, use .add()
       return True
```

```python
class Solution:
    def isHappy(self, n: int) -> bool:        
        record = set()

        while True:
            n = self.get_sum(n)
            if n == 1:
                return True
            
            # 如果中间结果重复出现，说明陷入死循环了，该数不是快乐数
            if n in record:
                return False
            else:
                record.add(n)

    def get_sum(self,n: int) -> int: 
        new_num = 0
        while n:
            n, r = divmod(n, 10)
            new_num += r ** 2
        return new_num
```



in C++

```c++
class Solution {
public:
    int get_sum(int n){
        int sum = 0;
        while(n > 0){
            n = n / 10;
            sum += (n % 10) * (n % 10);
        }
        return sum;
    }

    bool isHappy(int n) {
        unordered_set<int> record;

        while(1){
            n = get_sum(n);
            if(n == 1){ return true; };
            if(record.find(n) != record.end()){
                return false;
            } else{
                record.insert(n);
            }
        }
    }
};
```





#### 2. Sum of two numbers

Given an array of integers `nums` and an integer `target`, return *indices of the two numbers such that they add up to `target`*.

You may assume that each input would have ***exactly\* one solution**, and you may not use the *same* element twice.

You can return the answer in any order.

https://leetcode.cn/problems/two-sum/description/



- **O (n)**

==enumerate can get index and value at the same time,  !!! key and value==

Every time we iterate to a number, we want to know whether a previous number can be added to the target, target - num in previous. so we need to store index and previous value, and we use map.

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = set()

        for index, num in enumerate(nums):
            if (target - num) in seen:
                return [nums.index(target - num), index]
            seen.add(num)
```

**list.index(value) : return index of this value**

**add value in set : set.add(number)**



==**The double pointer method cannot be used**==, because:

The sum of two numbers requires <u>the return of the index, and the double pointer method must be sorted. Once sorted, the index of the original array is changed.</u>

If The sum of two numbers requires the return of the value, the double pointer method can be used.



In C++:

we need key-value, but unordered. now the value is the index we need, and we use the key to do the calculation.

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, int> seen;
        for(int i; i < nums.size() ; ++i){
            auto index = seen.find(target - nums[i]);
            if(index != seen.end()){
                return {index->second, i};
            }
            seen.insert(pair<int, int>(nums[i], i));
        }
        return {};
    }
};
```





#### 454. 4Sum II

Given four integer arrays `nums1`, `nums2`, `nums3`, and `nums4` all of length `n`, return the number of tuples `(i, j, k, l)` such that:

- `0 <= i, j, k, l < n`
- `nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0`

https://leetcode.cn/problems/4sum-ii/



- **O (n * n)**

==for dictionary, dic.get(index, 0) will return the value or return 0 if not exists.==

```python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        partSum = dict()

        for i in nums1:
            for j in nums2:
                partSum[i + j] = partSum.get(i+j, 0) + 1
        
        counts = 0
        for i in nums3:
            for j in nums4:
                if -(i + j) in partSum:
                    counts += partSum[-(i + j)]
        
        return counts
```

sometimes partSum[- (i + j) ] may be not exist, so we use if -(i + j) in partSum.



In C++:

```c++
class Solution {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int, int> umap; //key:a+b，value:a+b counts
        for (int a : A) {
            for (int b : B) {
                umap[a + b]++;
            }
        }
      
        int count = 0; 
        for (int c : C) {
            for (int d : D) {
                if (umap.find(0 - (c + d)) != umap.end()) {
                    count += umap[0 - (c + d)];
                }
            }
        }
        return count;
    }
};
```





#### 383. Ransom Notes

Given two strings `ransomNote` and `magazine`, return `true` *if* `ransomNote` *can be constructed by using the letters from* `magazine` *and* `false` *otherwise*.

Each letter in `magazine` can only be used once in `ransomNote`.

https://leetcode.cn/problems/ransom-note/description/



- **O(n)**

Easy method: for string, it has **.count('a') can count frequency of letter**

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        for char in ransomNote:
            if char in magazine and ransomNote.count(char) <= magazine.count(char):
                continue
            else:
                return False
        return True
```

==If iterate all letter and return nothing false, then we return True.==



```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        count_ltr = [0] * 26
        for ltr in magazine:
            count_ltr[ord(ltr) - ord('a')] += 1
        
        for ltr in ransomNote:
            count_ltr[ord(ltr) - ord('a')] -= 1
            if count_ltr[ord(ltr) - ord('a')] < 0:
                return False
        
        return True
```



```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        count_magazine = dict()
        for letter in magazine:
            count_magazine[letter] = count_magazine.get(letter, 0) + 1
        
        for letter in ransomNote:
            if letter not in count_magazine or count_magazine[letter] == 0:
                return False
						count_magazine[letter] -= 1
            
        return True
```

**For large data, map will need more memory space than array, so we consider array first.**



In C++:

```c++
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        vector<int> count_ltr(26, 0);
        for(int i = 0; i < magazine.size(); i++){
            count_ltr[magazine[i] - 'a']++;
        }
        for(int i = 0; i < ransomNote.size(); i++){
            count_ltr[ransomNote[i] - 'a']--;
            if(count_ltr[ransomNote[i] - 'a'] < 0){
                return false;
            }
        }
        return true;
    }
};
```





### [15. 3Sum](https://leetcode.cn/problems/3sum/description/)

Given an integer array nums, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.



#### ==Double-Pointers==

1. Sort the numbers, if nums[0] > 0, empty result.

2. We have three pointers, one is to do the for-loop. 

==The second is the i + 1 position, the third is the last one. These two are to do the Related-Pointer Method, the end condition is left = right==

3. If nums[i] + nums[left] + nums[right] > 0, we need to move the right one to left to decrease the sum.

If nums[i] + nums[left] + nums[right] < 0, we need to move the left one to right to increase the sum.

![15.三数之和](https://camo.githubusercontent.com/195c8b882887df8176a7fc49b069b8a2696f1f47c962a9ebdc3fe961db473503/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f676966732f31352e2545342542382538392545362539352542302545342542392538422545352539322538432e676966)

4. ==We need to remove duplicates, so skip i = i - 1, and for left pointer, skip left = left + 1.== 

We can also skip right = right - 1, because for the same right value and i, there is only one corresponding left value to make sum = 0. If we skip duplicates, then while-loop will decrease to save memory.



#### Solution

==O(n ^ 2) / O(1)==

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        
        for i in range(len(nums)):
            if nums[i] > 0:
                return result
            
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
            left = i + 1
            right = len(nums) - 1
            
            while right > left:
                sum_ = nums[i] + nums[left] + nums[right]
                
                if sum_ < 0:
                    left += 1
                elif sum_ > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # 跳过相同的元素以避免重复
                    while right > left and nums[left] == nums[left + 1]:
                        left += 1
                    while right > left and nums[right] == nums[right - 1]:
                        right -= 1
                        
                    right -= 1
                    left += 1
                    
        return result
```



- ==**Hash table( O(n^2) / O(n)**==  ) (difficult for duplicates)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()

        for i in range(len(nums)):
            if nums[i] > 0:
                return result
            if nums[i] == nums[i - 1] and i > 0:
                continue
            partSum = {}
            for j in range(i+1, len(nums)):
                if j > i + 2 and nums[j] == nums[j - 1] == nums[j - 2]:
                    continue
                c = 0 - nums[i] - nums[j]
                if c in partSum:
                    result.append([nums[i], nums[j], c])
                    partSum.pop(c)
                else:
                    partSum[nums[j]] = j
        
        return result
```



- In C++:  double-pointers and hash table

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());

        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] > 0) {
                return result;
            }
            // 错误去重a方法，将会漏掉-1,-1,2 这种情况
            /*
            if (nums[i] == nums[i + 1]) {
                continue;
            }
            */
            // 正确去重a方法
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.size() - 1;
            while (right > left) {
                // 去重复逻辑如果放在这里，0，0，0 的情况，可能直接导致 right<=left 了，从而漏掉了 0,0,0 这种三元组
                /*
                while (right > left && nums[right] == nums[right - 1]) right--;
                while (right > left && nums[left] == nums[left + 1]) left++;
                */
                if (nums[i] + nums[left] + nums[right] > 0) right--;
                else if (nums[i] + nums[left] + nums[right] < 0) left++;
                else {
                    result.push_back(vector<int>{nums[i], nums[left], nums[right]});
                    // 去重逻辑应该放在找到一个三元组之后，对b 和 c去重
                    while (right > left && nums[right] == nums[right - 1]) right--;
                    while (right > left && nums[left] == nums[left + 1]) left++;

                    // 找到答案时，双指针同时收缩
                    right--;
                    left++;
                }
            }

        }
        return result;
    }
};
```

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        // 找出a + b + c = 0
        // a = nums[i], b = nums[left], c = nums[right]
        for (int i = 0; i < nums.size(); i++) {
            // 排序之后如果第一个元素已经大于零，那么无论如何组合都不可能凑成三元组，直接返回结果就可以了
            if (nums[i] > 0) {
                return result;
            }
            // 错误去重a方法，将会漏掉-1,-1,2 这种情况
            /*
            if (nums[i] == nums[i + 1]) {
                continue;
            }
            */
            // 正确去重a方法
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.size() - 1;
            while (right > left) {
                // 去重复逻辑如果放在这里，0，0，0 的情况，可能直接导致 right<=left 了，从而漏掉了 0,0,0 这种三元组
                /*
                while (right > left && nums[right] == nums[right - 1]) right--;
                while (right > left && nums[left] == nums[left + 1]) left++;
                */
                if (nums[i] + nums[left] + nums[right] > 0) right--;
                else if (nums[i] + nums[left] + nums[right] < 0) left++;
                else {
                    result.push_back(vector<int>{nums[i], nums[left], nums[right]});
                    // 去重逻辑应该放在找到一个三元组之后，对b 和 c去重
                    while (right > left && nums[right] == nums[right - 1]) right--;
                    while (right > left && nums[left] == nums[left + 1]) left++;

                    // 找到答案时，双指针同时收缩
                    right--;
                    left++;
                }
            }

        }
        return result;
    }
};
```



#### Delete duplicates

![image-20250209153944417](/Users/annahuang/Library/Application Support/typora-user-images/image-20250209153944417.png)

![image-20250209154008015](/Users/annahuang/Library/Application Support/typora-user-images/image-20250209154008015.png)





### [18. 4Sum](https://leetcode.cn/problems/4sum/)

Given an array `nums` of `n` integers, return *an array of all the **unique** quadruplets* `[nums[a], nums[b], nums[c], nums[d]]` such that:

- `0 <= a, b, c, d < n`
- `a`, `b`, `c`, and `d` are **distinct**.
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

You may return the answer in **any order**.



#### Idea

==4sum and 3sum are the same idea, both use the double pointer method, just add one more for-loop.==



1. ==**Sort it, if num[i] >= 0  and  nums[i] > target,  break**.==

nums[i] < 0, target > 0 : No break.  nums[i] < 0, target < 0 : not sure, no break.

nums[i] > 0, target > 0 : can break. nums[i] > 0, target < 0 : can break

![image-20250220101352052](/Users/annahuang/Library/Application Support/typora-user-images/image-20250220101352052.png)



2. for the second loop, consider the sum of first two numbers and target.

​	==if nums[i] + nums[j] > target and target > 0==



3. **unique** quadruplets: the four values cannot be duplicates.

​	for pointer 1, remove nums[i] = num[i - 1], then must i > 0.

​	for pointer 2, remove nums[j] = num[j - 1], then must j > i + 1.

​	for pointer 3, remove nums[left] == nums[left + 1], then must left < right.

​	for pointer 4, remove nums[right] == nums[right - 1], then must left < right.



3. a, b, c, d, distinct: no same index 

​	for pointer 2, different from pointer 1, so range (i+1, n-1)

​	for pointer 3, different from pointer 2, so j + 1.



#### Solution

==**O(n^3) /  O(1)**==

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        result = []
        for i in range(n):
            if nums[i] > target and nums[i] >= 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i+1, n):
                if nums[i] + nums[j] > target and target > 0:
                    break
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                left, right = j + 1, n - 1
                while left < right:
                    s = nums[i] + nums[j] + nums[left] + nums[right]
                    if  s == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif s < target:
                        left = left + 1
                    else:
                        right = right - 1
        return result
```



- **by dictionary**

```python
class Solution(object):
    def fourSum(self, nums, target):
        # 创建一个字典来存储输入列表中每个数字的频率
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # 创建一个集合来存储最终答案，并遍历4个数字的所有唯一组合
        ans = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    val = target - (nums[i] + nums[j] + nums[k])
                    if val in freq:
                        # 确保没有重复
                        count = (nums[i] == val) + (nums[j] == val) + (nums[k] == val)
                        if freq[val] > count:
                            ans.add(tuple(sorted([nums[i], nums[j], nums[k], val])))
        
        return [list(x) for x in ans]

```



- in C++:

```c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        for (int k = 0; k < nums.size(); k++) {
            // 剪枝处理
            if (nums[k] > target && nums[k] >= 0) {
            	break; // 这里使用break，统一通过最后的return返回
            }
            // 对nums[k]去重
            if (k > 0 && nums[k] == nums[k - 1]) {
                continue;
            }
            for (int i = k + 1; i < nums.size(); i++) {
                // 2级剪枝处理
                if (nums[k] + nums[i] > target && nums[k] + nums[i] >= 0) {
                    break;
                }

                // 对nums[i]去重
                if (i > k + 1 && nums[i] == nums[i - 1]) {
                    continue;
                }
                int left = i + 1;
                int right = nums.size() - 1;
                while (right > left) {
                    // nums[k] + nums[i] + nums[left] + nums[right] > target 会溢出
                    if ((long) nums[k] + nums[i] + nums[left] + nums[right] > target) {
                        right--;
                    // nums[k] + nums[i] + nums[left] + nums[right] < target 会溢出
                    } else if ((long) nums[k] + nums[i] + nums[left] + nums[right]  < target) {
                        left++;
                    } else {
                        result.push_back(vector<int>{nums[k], nums[i], nums[left], nums[right]});
                        // 对nums[left]和nums[right]去重
                        while (right > left && nums[right] == nums[right - 1]) right--;
                        while (right > left && nums[left] == nums[left + 1]) left++;

                        // 找到答案时，双指针同时收缩
                        right--;
                        left++;
                    }
                }

            }
        }
        return result;
    }
};
```





### Summary

[Hash Table Summary](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/%E5%93%88%E5%B8%8C%E8%A1%A8%E6%80%BB%E7%BB%93.md)







## 4. Strings

### [344 Reverse String](https://leetcode.cn/problems/reverse-string/description/)

Write a function that reverses a string. The input string is given as an array of characters `s`.

You must do this by modifying the input array [in-place](https://en.wikipedia.org/wiki/In-place_algorithm) with `O(1)` extra memory.



#### ==**Double Pointers Method**==

Define two pointers, one is from the left side, another is from the tail side, exchange character and move towards the middle.

<img src="https://camo.githubusercontent.com/5bc12f52464405ddf17bf262c5d9f0da5fd7a1f7afe601779a62ef19ebe7a78f/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f676966732f3334342e2545352538462538442545382542442541432545352541442539372545372541432541362545342542382542322e676966" alt="344.反转字符串" style="zoom:80%;" />

**Notes: If the key part of the question can be solved directly using library functions, it is recommended not to use library functions.**

If the library function is only a small part of the problem-solving process and you are already very clear about the internal implementation of the library function, you can consider using the library function.



#### Solution

==O(n) / O(1)==

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        i = 0
        j = len(s) - 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
```

```python
# method 2: Library function
	s[:] = reversed(s)
  
# method 3: Slice
  s[:] = s[::-1]

# method 4: self method
	s.reverse()

# method 5: Derived List
	s[:] = [s[i] for i in range(len(s) - 1, -1, -1)]
  
# method 6: range
	n = len(s)
  for i in range(n // 2):
    s[i], s[n - i - 1] = s[n - i - 1], s[i]
 
```

```c++
class Solution {
public:
    void reverseString(vector<char>& s) {
        for(int i = 0, j = s.size() - 1; i < j; ++i, --j){
            swap(s[i], s[j]);
        }
    }
};
```





### [541. Reverse String II](https://leetcode.cn/problems/reverse-string-ii/)

Given a string `s` and an integer `k`, reverse the first `k` characters for every `2k` characters counting from the start of the string.

If there are fewer than `k` characters left, reverse all of them. If there are less than `2k` but greater than or equal to `k` characters, then reverse the first `k` characters and leave the other as original.



#### Fixed Rule in For-Loop

In fact, <u>when traversing the string, we only need to let i += (2 * k), i moves 2 * k each time, and then determine whether we need to reverse interval.</u>

So ==when you need to process a string segment by segment according to a fixed rule, you should think about doing something with the for loop expression==.



#### Solution

**Reversing**: `[::-1]` reverses the substring obtained from the slicing operation.

**convert string to an array of character** ---> `list(str)`

**convert an array of character to string** ---> ``''.join(arr)``

==O(n) / O(1)==

```python
class Solution:
    def reverse(self, text):
        i, j = 0, len(text) - 1
        while i < j:
            text[i], text[j] = text[j], text[i]
            i += 1
            j -= 1
        return text

    def reverseStr(self, s: str, k: int) -> str:
        s = list(s)
        
        for cur in range(0, len(s), 2*k):
            s[cur:cur + k] = self.reverse(s[cur:cur + k])

        return ''.join(s)
```

```py
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        # Two pointers. Another is inside the loop.
        p = 0
        while p < len(s):
            p2 = p + k
            # Written in this could be more pythonic.
            s = s[:p] + s[p: p2][::-1] + s[p2:]
            p = p + 2 * k
        return s
```

```c++
class Solution {
public:
    void reverse(string& s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            swap(s[i], s[j]);
        }
    }
    string reverseStr(string s, int k) {
        for (int i = 0; i < s.size(); i += (2 * k)) {
            // 剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符
            if (i + k <= s.size()) {
                reverse(s, i, i + k - 1);
                continue;
            }
            // 剩余字符少于 k 个，则将剩余字符全部反转。
            reverse(s, i, s.size() - 1);
        }
        return s;
    }
};
```

==**reverse**:== library function in C++ and need to include algorithms. eg. reverse(s.begin() + i, s.begin() + i + k)

**==swap==**: swap two values.  eg. swap(s[i], s[j])





### [151. Reverse Words in a String](https://leetcode.cn/problems/reverse-words-in-a-string/)

Given an input string `s`, reverse the order of the **words**.

A **word** is defined as a sequence of non-space characters. The **words** in `s` will be separated by at least one space.

Return *a string of the words in reverse order concatenated by a single space.*

**Note** that `s` may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.



#### Ideas

1. Split words by any space, then reverse words, finally combine it to string.

2. Split words, then reverse words by double pointers method, finally combine.
3. reverse the whole string, then split it and reverse every word, finally combine.

4. erase surplus spaces(double pointers method), reverse the whole string, then reverse every word.



#### Solutions

==O(n) / O(n)==

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        words = s.split()
        words = words[::-1]
        return ' '.join(words)
```

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        words = s.split()

        # 反转单词
        left, right = 0, len(words) - 1
        while left < right:
            words[left], words[right] = words[right], words[left]
            left += 1
            right -= 1

        return " ".join(words)
```

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # 反转整个字符串
        s = s[::-1]
        # 将字符串拆分为单词，并反转每个单词
        s = ' '.join(word[::-1] for word in s.split())
        return s
```

==O(n) / O(1)==

```c++
class Solution {
public:
    void eraseSpace(string& s){
        int slow = 0;
        for(int i = 0; i < s.size(); ++i){
            if(s[i] != ' '){
                if(slow != 0 ){
                    s[slow] = ' ';
                    slow++;}
                while(s[i] != ' ' && i < s.size()){
                    s[slow] = s[i];
                    slow++;
                    i++;
                }
            }
        }
        s.resize(slow);
    }
    
    void reverseStr(string& s, int left, int right){
        for(; left < right; left++, right--){
            swap(s[left], s[right]);
        }
    }

    string reverseWords(string s) {
        eraseSpace(s);
        reverseStr(s, 0, s.size() - 1);
        int start = 0;
        for(int i = 0; i <= s.size(); ++i){
            if(i == s.size() || s[i] == ' '){
                reverseStr(s, start, i-1);
                start = i + 1;
            }
        }
        return s;
    }
};
```





### [Substitude Number](https://kamacoder.com/problempage.php?pid=1064)

Given a string s that contains lowercase letters and numeric character , write a function that leaves the alphabetic characters in the string unchanged and replaces each numeric character with `'number'`.

For example, given the input string "a1b2c3", the function should convert it to "anumberbnumbercnumber".



#### ==Filling from back to front==

Filling from front to back is an O(n^2) algorithm, because each time an element is added, all elements after the added element must be moved backward as a whole.

==In fact, many array filling problems are solved by first expanding the array to the filled size, and then operating from back to front.==



1. Expand the length of arrays to the size after substituding numeric characters with 'number'.

<img src="https://camo.githubusercontent.com/7c211fe78a479266716315848a7033862956fbc397c45c267ac8506f7591e12e/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303233313033303136353230312e706e67" alt="img" style="zoom:50%;" />

2. ==Substitude numeric characters with 'number' from tail to head. ( double pointers method )==

![img](https://camo.githubusercontent.com/68314925e7797b5a8fc2c92043e5df937e587d5d1fb3c5b15154473beb965223/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303233313033303137333035382e706e67)



#### Solution

==O(n) / O(1)==

```python
def SubstitudeNum(string):
		return ''.join(['number' if char.isdigit() else char for char in string])

if __name__ == "__main__":
		s = input()
	
		print(SubstitudeNum(s))
```

```python
#include <iostream>
using namespace std;
int main() {
    string s;
    
    while (cin >> s) {
        int oldIndex = s.size() - 1;
        
        int count = 0;
        for(int i = 0; i < s.size(); ++i){
            if(s[i] >= '0' && s[i] <= '9'){
                count++;
            }
        }
        s.resize(s.size() + count * 5);
        
        int newIndex = s.size() - 1;
        for(int i = oldIndex; i >= 0 ; i--){
            if(s[i] >= '0' && s[i] <= '9'){
                s[newIndex--] = 'r';
                s[newIndex--] = 'e';
                s[newIndex--] = 'b';
                s[newIndex--] = 'm';
                s[newIndex--] = 'u';
                s[newIndex--] = 'n';
            }else{
                s[newIndex--] = s[i];
            }
        }

        cout << s << endl;       
    }
}
```





### [Right-handed string](https://kamacoder.com/problempage.php?pid=1065)

The right rotation operation of a string is to move several characters at the end of the string to the front of the string. Given a string s and a positive integer k, please write a function to move the last k characters of the string to the front of the string to implement the right rotation operation of the string.



#### Part Reversion

Do not use additional space, just operate in itself.

==We can reverse the whole string, and reverse the first k-th,  and reverse the last string seperately.==

<img src="https://camo.githubusercontent.com/3bd5b6ba12fdd1c80278bec4938a3cf5982d3cafd1074ee190460d682460206e/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303233313130363137323035382e706e67" alt="img" style="zoom:50%;" />



#### Solution

```python
if __name__ == '__main__':
    k = int(input())
    s = input()
    
    s = list(s)
    s = s[len(s) - k:] + s[:len(s) - k]
    print(''.join(s))
```

```python
k = int(input())
s = input()

print(s[-k:] + s[:-k])
```

```c++
#include <iostream>
#include<algorithm>
using namespace std;

int main(){
    int k;
    string s;
    cin >> k;
    cin >> s;
    
    reverse(s.begin(), s.end());
    reverse(s.begin(), s.begin() + k);
    reverse(s.begin() + k, s.end());
    
    cout << s << endl;
}
```





### ==KMP Algorithm - match string==

The main idea of KMP is that when a string mismatch occurs, we can know part of the string that has been matched before, and use this information to avoid matching again from the beginning.

[帮你把KMP算法学个通透！（理论篇）](https://www.bilibili.com/video/BV1PD4y1o7nd/)

[求next数组代码篇](https://www.bilibili.com/video/BV1M5411j7Xx/)



#### ==next arrays = prefix table==

When the pattern string does not match the main string (text string), the prefix table records where the pattern string should start matching again.

**Record the length of the same prefix and suffix before (including) i-th character in the pattern string . **

Eg. Find a pattern string:  `aabaaf` in text string:  `aabaabaafa`

![KMP详解1](https://camo.githubusercontent.com/16837c00cb51cf8498a06958072b8fdc52691a0313ec275fd19fc3b23bbda720/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f676966732f4b4d50254537254232254245254538254145254232312e676966)

**下标5之前这部分的字符串（也就是字符串aabaa）的最长相等的前缀 和 后缀字符串是 子字符串aa ，因为找到了最长相等的前缀和后缀，匹配失败的位置是后缀子串的后面，那么我们找到与其相同的前缀的后面重新匹配就可以了。**

因为匹配失败的位置前面的字符都已经匹配成功了，那么匹配失败的位置前面几个字符是后缀(已匹配成功），如果前缀有跟这个后缀相同的，也相当于匹配上了，所以我们可以直接从这个前缀的后面一个字符开始继续匹配。

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20250224113653704.png" alt="image-20250224113653704" style="zoom:30%;" />



#### Longest equal prefix and suffix

Prefix includes all the continuous substring that starts with the first character but does not include the last character.

Suffix includes all the  continuous substring that ends with the last character but does not include the first character.

Eg. The longest equal prefix and suffix of string a is 0. 

The longest equal prefix and suffix of string aa is 1. (a and a)

The longest equal prefix and suffix of string aab is 0. 

The longest equal prefix and suffix of string aabaa is 2. (a and a, aa and aa)

The longest equal prefix and suffix of string aabaaf is 0. 

<img src="https://camo.githubusercontent.com/dc9faa71d661bbf55da1b119c5262e8964211dd52f60090af181a6d486dc5724/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f706963732f4b4d50254537254232254245254538254145254232382e706e67" alt="KMP精讲8" style="zoom:80%;" />

找到的不匹配的位置， 那么此时我们要看它的前一个字符的前缀表的数值是多少。

为什么要前一个字符的前缀表的数值呢，因为要找前面字符串的最长相同的前缀和后缀。



#### next arrays VS prefix table

next数组就可以是前缀表，但是很多实现都是把前缀表统一减一（右移一位，初始位置为-1）之后作为next数组。

The next array can be a prefix table.

==Many implementations reduce the prefix table by one (shift right one position, with the initial position being -1) and use it as the next array.==

Example for reducing prefix table by one:

![KMP精讲4](https://camo.githubusercontent.com/21598bda446d96cf3247fbc0090624e809f876b16f6ca38d431abd9671d7bee4/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f676966732f4b4d50254537254232254245254538254145254232342e676966)

#### Time Complexity

n is the length of the text string and m is the length of the pattern string.

The next array must be generated separately, and the time complexity is O(m).

Because the matching position is constantly adjusted according to the prefix table during the matching process, it can be seen that the matching process is O(n).

Therefore, **==the time complexity of the entire KMP algorithm is O(n+m).==**



#### Construct next arrays

1. 初始化：

定义两个指针i和j，==j指向前缀末尾位置，i指向后缀末尾位置==, 然后还要对next数组进行初始化赋值.

next[i] 表示 i（包括i）之前最长相等的前后缀长度（其实就是j）, 所以初始化next[0] = j 。



2. 处理前后缀不相同的情况

因为j初始化为0，那么i就从1开始，进行s[i] 与 s[j]的比较。

所以遍历模式串s的循环下标i 要从 1开始，代码：for (int i = 1; i < s.size(); i++) 

如果 s[i] 与 s[j]不相同，也就是遇到 前后缀末尾不相同的情况，就要向前回退。

next[j]就是记录着j（包括j）之前的子串的相同前后缀的长度。

那么 s[i] 与 s[j] 不相同，就要找 j前一个元素在next数组里的值（就是next[j - 1]）。

如果回退之后仍然不匹配，就要继续回退。

回退的边界条件是j >=0。



3. 处理前后缀相同的情况

如果 s[i] 与 s[j] 相同，那么就同时向后移动i （for循环里）和j （增加操作），因为说明此时有相同的前后缀要继续往后匹配，同时还要将j（前缀的长度）赋给next[i], 因为next[i]要记录相同前后缀的长度。

```python
    void getNext(int* next, const string& s) {
        int j = 0;
        next[0] = 0;
        for(int i = 1; i < s.size(); i++) {
            while (j > 0 && s[i] != s[j]) { // j要保证大于0，因为下面有取j-1作为数组下标的操作
                j = next[j - 1]; // 注意这里，是要找前一位的对应的回退位置了
            }
            if (s[i] == s[j]) {
                j++;
            }
            next[i] = j;
        }
    }
```

![KMP精讲3](https://camo.githubusercontent.com/98a08768ee993164cde581887c338296260f62b50081fdaa28e9208677c1e9f9/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f676966732f4b4d50254537254232254245254538254145254232332e676966)





### [ 28. Find the Index of the First Occurrence in a String](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)

Given two strings `needle` and `haystack`, return the index of the first occurrence of `needle` in `haystack`, or `-1` if `needle` is not part of `haystack`.



#### match by next arrays

Define two pointers, j for the start of the pattern string, i for the start of the text string.

If s[i] cannot match t[j], j will find the next position to match in next arrays.

If j points to the end of the next string, matching successful



#### Solution

==O(n + m) / O(m)==

```python
class Solution:
    def getNext(self, next, s):
        j = 0
        next.append(0)
        for i in range(1, len(s)):
            while j > 0 and s[j] != s[i]:
                j = next[j - 1]
            if s[j] == s[i]:
                j += 1
            next.append(j)


    def strStr(self, haystack: str, needle: str) -> int:
        if len(needle) == 0:
            return 0
        next = []
        self.getNext(next, needle)

        j = 0
        for i in range(len(haystack)):
            while j > 0 and haystack[i] != needle[j]:
                j = next[j - 1]
            if needle[j] == haystack[i]:
                j += 1
            if j == len(needle):
                return (i - len(needle) + 1)

        return -1
```

```python
# simple but rode solution

		return haystack.find(needle)

		try:
         return haystack.index(needle)
    except ValueError:
         return -1
```

```c++
class Solution {
public:
    void getNext(int* next, const string& s) {
        int j = 0;
        next[0] = 0;
        for(int i = 1; i < s.size(); i++) {
            while (j > 0 && s[i] != s[j]) {
                j = next[j - 1];
            }
            if (s[i] == s[j]) {
                j++;
            }
            next[i] = j;
        }
    }
    int strStr(string haystack, string needle) {
        if (needle.size() == 0) {
            return 0;
        }
        vector<int> next(needle.size());
        getNext(&next[0], needle);
        int j = 0;
        for (int i = 0; i < haystack.size(); i++) {
            while(j > 0 && haystack[i] != needle[j]) {
                j = next[j - 1];
            }
            if (haystack[i] == needle[j]) {
                j++;
            }
            if (j == needle.size() ) {
                return (i - needle.size() + 1);
            }
        }
        return -1;
    }
};
```





### [459. Repeated Substring Pattern](https://leetcode.cn/problems/repeated-substring-pattern/description/)

Given a string `s`, check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.



#### ==KMP Algorithm==

**==如果这个字符串s是由重复子串组成，那么最长相等前后缀不包含的子串是字符串s的最小重复子串==。**

<img src="/Users/annahuang/Library/Application Support/typora-user-images/image-20250304101428940.png" alt="image-20250304101428940" style="zoom:150%;" />

<img src="https://camo.githubusercontent.com/c13fe6984f293409f2f8461e1b281c6915564d65c02e14327ac9a5db0ba09ab9/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303234303931333131303834312e706e67" alt="img" style="zoom:65%;" />



1. **Important Analysis** 

next 数组记录的就是最长相同前后缀， 如果 `next[len - 1] != 0`，则说明字符串有最长相同的前后缀（就是字符串里的前缀子串和后缀子串相同的最长长度）。

最长相等前后缀的长度为：`next[len - 1]`。数组长度为：len。

`len - next[len - 1]` 是最长相等前后缀不包含的子串的长度。

如果`len % (len - next[len - 1] == 0` ，则说明数组的长度正好可以被 "最长相等前后缀不包含的子串的长度" 整除，说明该字符串有重复的子字符串。



2. Example

![459.重复的子字符串_1](https://camo.githubusercontent.com/0d2bac3cf5d4a2d2c5ba700ae4430df6cdfe452417e955db5a6f704f3aa5943b/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f706963732f3435392e2545392538372538442545352541342538442545372539412538342545352541442539302545352541442539372545372541432541362545342542382542325f312e706e67)

`next[len - 1] = 7`，`next[len - 1] + 1 = 8`，8就是此时字符串asdfasdfasdf的最长相同前后缀的长度。

`(len - (next[len - 1] + 1))` 也就是： 12(字符串的长度) - 8(最长公共前后缀的长度) = 4， 为最长相同前后缀不包含的子串长度

4可以被 12(字符串的长度) 整除，所以说明有重复的子字符串（asdf）。



#### Mobile Matching

When the internal of a string is constituted by duplicated substrings, then the constructure must be:

<img src="https://camo.githubusercontent.com/ad810b4a089f271ba15f86357bf6c1c0426691d7bc409dcaa944c72e91ba13e9/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303232303732383130343531382e706e67" alt="图一" style="zoom:67%;" />

Since there is the same substring in the front and the back, we combine "string + string", the the latter and the former substring will definitely form a string.

![图二](https://camo.githubusercontent.com/1e360eb7ba304aa4c3f9a44f7850981e6068139d3483e4ffbb33ca289310e0fb/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303232303732383130343933312e706e67)

Notes: we need to delete the first and the last character of "s + s", to avoid getting the original string.



#### Solution

==O(n) / O(n)==

```python
class Solution(object):
    def getNext(self, next, string):
        next[0] = 0
        j = 0
        for i in range(1, len(string)):
            while j > 0 and string[i] != string[j]:
                j = next[j - 1]
            if string[i] == string[j]:
                j += 1
            next[i] = j

    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) == 0:
            return False
        
        next = [0] * len(s)
        self.getNext(next, s)
        length = len(s)
        if next[length - 1] != 0 and length % (length - next[length - 1]) == 0:
            return True
        return False
```

```c++
class Solution {
public:
    void getNext(int* next, string& s){
        next[0] = 0;
        int j = 0;
        for(int i = 1; i < s.size(); i++){
            while(j > 0 && s[i] != s[j]){
                j = next[j - 1];
            }
            if(s[i] == s[j]){
                j++;
            }
            next[i] = j;
        }
    }

    bool repeatedSubstringPattern(string s) {
        if(s.size() == 0){
            return false;
        }
        int next[s.size()];
        getNext(next, s);
        int len = s.size();
        if(next[len - 1] != 0 && len % (len - next[len - 1]) == 0 ){
            return true;
        }
        return false;
    }
};
```



==O(n) / O(1)==

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        if n <= 1:
            return False
        ss = s[1:] + s[:-1] 
        print(ss.find(s))              
        return ss.find(s) != -1
```

```c++
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        string t = s + s;
        t.erase(t.begin()); 
        t.erase(t.end() - 1); // 掐头去尾
        if (t.find(s) != std::string::npos) 
          	return true; // r
        return false;
    }
};
```





### [Summary](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/%E5%AD%97%E7%AC%A6%E4%B8%B2%E6%80%BB%E7%BB%93.md)

**==Double-Pointers Method / KMP Algorithms==**

#### Most common string library functions

1. **len(s)**
   
   **Time Complexity**: O(1)

3. **s1 + s2**
   
   **Time Complexity**: O(len(s1) + len(s2))

4. **s * n**
   
   - **Description**: Repeats the string `s` `n` times.
   - **Time Complexity**: O(len(s) * n)
   
5. **s.find(sub)**
   
   - **Description**: Returns the lowest index in `s` where substring `sub` is found.
   - **Time Complexity**: O(len(s) * len(sub))
   
6. **s.index(sub)**
   - **Description**: Similar to `find()`, but raises a `ValueError` if `sub` is not found.
   - **Time Complexity**: O(len(s) * len(sub))

7. **s.count(sub)**
   - **Description**: Returns the number of non-overlapping occurrences of substring `sub` in `s`.
   - **Time Complexity**: O(len(s) * len(sub))

8. **s.lower()**
   
   - **Description**: Returns a copy of the string `s` with all characters converted to lowercase.
   - **Time Complexity**: O(len(s))
   
9. **s.upper()**
   
   - **Description**: Returns a copy of the string `s` with all characters converted to uppercase.
   - **Time Complexity**: O(len(s))
   
10. **s.replace(old, new)**
    - **Description**: Returns a copy of the string `s` with all occurrences of substring `old` replaced by `new`.
    - **Time Complexity**: O(len(s) * len(old))

11. **s.split(sep)**
    - **Description**: Splits the string `s` into a list of substrings based on the separator `sep`.
    - **Time Complexity**: O(len(s))

12. **s.strip()**
    - **Description**: Returns a copy of the string `s` with leading and trailing whitespace removed.
    - **Time Complexity**: O(len(s))

13. **s.join(iterable)**
    - **Description**: Concatenates the strings in the iterable `iterable` with `s` as the separator.
    - **Time Complexity**: O(len(s) * len(iterable))

13. reversed(s)

- **Time Complexity**: O( n )







## 5. Double Pointers Method

### [27. Remove Element](https://leetcode.cn/problems/remove-element/)



### [344 Reverse String](https://leetcode.cn/problems/reverse-string/description/)



### [Substitude Number](https://kamacoder.com/problempage.php?pid=1064)



### [151. Reverse Words in a String](https://leetcode.cn/problems/reverse-words-in-a-string/)



### [206. Reverse Linked List](https://leetcode.cn/problems/reverse-linked-list/description/)



### [160. Intersection of Two Linked Lists](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/)



### [15. 3Sum](https://leetcode.cn/problems/3sum/description/)



### [18. 4Sum](https://leetcode.cn/problems/4sum/)
