#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import deque

def dfs(graph, start, goal):
    visited = set() # Các nút đã được duyệt
    trace = {} # Lưu vết đường đi
    stack = deque() # Tập biên O
    temp = deque() # Tác dụng hiển thị tập O + lưu vểt
    stack.append(start)
    temp.append(start)
    trace[start] = 0
    print("STT\tNút được mở rộng\tTập biên O")
    stt = 0
    print(stt, list(stack), sep = '\t\t\t\t')

    while stack:
        stt += 1
        node = stack.popleft() # Lấy đỉnh stack rồi loại bỏ đỉnh đó ra khỏi stack
        node_temp = temp.popleft() # Tác dụng hiển thị
        
        if node not in visited: # Kiểm tra đỉnh đang xét đã được duyệt chưa
            visited.add(node)

            if node == goal: # Đỉnh đang xét là đích
                print(stt, node_temp, "Đích", sep='\t\t')   
                break
            
            for neighbor in graph[node][::-1]: # Duyệt các đỉnh kề với đỉnh đang xét
                if neighbor not in visited: # Kiểm tra các đỉnh đó đã được duyệt chưa
                    stack.appendleft(neighbor) 
                    temp.appendleft(neighbor+"("+node+")")
                    trace[neighbor] = node # Lưu vết
                    
        print(stt, node_temp, list(temp), sep='\t\t') # In trạng thái   
    
    # In đường đi
    print("Đường đi: ", end = "")
    print(goal, end = "")
    while trace[goal] != 0:
        print(" <-", trace[goal], end = "")
        goal = trace[goal]


# In[2]:


from queue import Queue

def bfs(graph, start, goal):
    visited = set() # Các nút đã được duyệt
    trace = {} # Lưu vết đường đi
    q = Queue() # Tập biên O
    temp = Queue() # Tác dụng hiển thị tập O + lưu vểt
    q.put(start)
    temp.put(start)
    trace[start] = 0
    print("STT\tNút được mở rộng\tTập biên O")
    stt = 0
    print(stt, list(q.queue), sep = '\t\t\t\t')

    while not q.empty():
        stt += 1
        node = q.get() # Lấy và loại bỏ đỉnh ở đầu hàng đợi
        node_temp = temp.get() # Tác dụng hiển thị
        
        if node not in visited:
            visited.add(node)

            if node == goal:
                print(stt, node_temp, "Đích", sep='\t\t')   
                break
            
            for neighbor in graph[node]: # Duyệt các đỉnh kề với đỉnh đang xét
                # Kiểm tra các đỉnh đó đã được duyệt chưa và có thuộc tập O không
                if neighbor not in visited and neighbor not in list(q.queue): 
                    q.put(neighbor) 
                    temp.put(neighbor+"("+node+")")
                    trace[neighbor] = node # Lưu vết
                    
        print(stt, node_temp, list(temp.queue), sep='\t\t')  
            
    # In đường đi
    print("Đường đi: ", end = "")
    print(goal, end = "")
    while trace[goal] != 0:
        print(" <-", trace[goal], end = "")
        goal = trace[goal]        


# In[3]:


from collections import deque

def ids(graph, start, goal):
    trace = {} # Lưu vết đường đi
    trace[start] = 0
    def dfs(graph, start, goal, depth, trace):
        visited = set() # Các nút đã được duyệt
        stack = deque() # Tập biên O
        temp = deque() # Tác dụng hiển thị tập O + lưu vểt
        stack.append(start)
        temp.append(start)
        i = 0
        stt = 0
        print("i =", depth)
        print("STT\tNút được mở rộng\tTập biên O")
        print(stt, list(stack), sep = '\t\t\t\t')  
        
        check = False
        while stack:
            if i == depth: # Nếu đến độ sâu tối đa rồi thì chỉ xét các đỉnh còn lại trong stack chứ không đào tiếp
                check = True
                
            stt += 1
            node = stack.popleft() # Lấy đỉnh stack rồi loại bỏ đỉnh đó ra khỏi stack
            node_temp = temp.popleft() # Tác dụng hiển thị

            if node not in visited: # Kiểm tra đỉnh đã được duyệt chưa
                visited.add(node)

                if node == goal: # Đỉnh đang xét là đích
                    print(stt, node_temp, "Đích", sep='\t\t')   
                    return True 

                if check is False: # Chưa đạt độ sâu tối đa
                    t = False # Kiểm tra đỉnh đó có neighbor không
                    for neighbor in graph[node][::-1]: # Duyệt các đỉnh kề với đỉnh đang xét
                        if neighbor not in visited:
                            t = True
                            stack.appendleft(neighbor) 
                            temp.appendleft(neighbor+"("+node+")")
                            trace[neighbor] = node # Lưu vết
                    if t is True: # Có neighbor thì tăng độ sâu
                        i += 1

            print(stt, node_temp, list(temp), sep='\t\t') # In trạng thái
        return False       
    
    # IDS search + hiển thị
    depth = 0
    flag = dfs(graph, start, goal, depth, trace)
    print("\n==========================================================\n")
    while(flag is False):
        depth += 1  
        flag = dfs(graph, start, goal, depth, trace)
        print("\n==========================================================\n")
        
    # In đường đi
    print("Đường đi: ", end = "")
    print(goal, end = "")
    while trace[goal] != 0:
        print(" <-", trace[goal], end = "")
        goal = trace[goal]


# In[4]:


class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}

    def neighbors(self, node):
        return self.edges[node]

    def get_cost(self, from_node, to_node):
        return self.weights[(from_node + to_node)]


# In[5]:


from queue import PriorityQueue

def ucs_weight(from_node, to_node, weights=None):
    return weights.get((from_node, to_node), 10e100) if weights else 1

def ucs(graph, start, goal):
    visited = set()
    queue = PriorityQueue()
    queue.put((0, start))

    while queue:
        cost, node = queue.get()
        if node not in visited:
            visited.add(node)

            if node == goal:
                return
            
            for i in graph.neighbors(node):
                if i not in visited:
                    total_cost = cost + graph.get_cost(node, i)
                    queue.put((total_cost, i))


# In[6]:


graph1 = {
        'A': ['B', 'C', 'D', 'E', 'Y'],
        'B': ['A', 'F', 'G'],
        'C': ['A', 'H'],
        'D': ['A', 'I', 'J'],
        'E': ['A', 'K', 'L'],
        'F': ['B', 'M', 'N', 'O'],
        'G': ['B', 'P', 'Q', 'R'],
        'H': ['C', 'S'],
        'I': ['D'],
        'J': ['D', 'T', 'U'],
        'K': ['E'],
        'L': ['E', 'V'],
        'M': ['F'],
        'N': ['F'],
        'O': ['F'],
        'P': ['G'],
        'Q': ['G'],
        'R': ['G'],
        'S': ['H', 'W', 'X'],
        'T': ['J'],
        'U': ['J', 'Y', 'Z'],
        'V': ['L'],
        'W': ['S'],
        'X': ['S'],
        'Y': ['U'],
        'Z': ['U']
}

graph2 = {
        'A': ['B', 'C', 'D', 'E'],
        'B': ['A', 'F', 'G', 'H'],
        'C': ['A', 'H'],
        'D': ['A', 'I', 'J'],
        'E': ['A', 'K', 'L'],
        'F': ['B', 'G', 'M', 'N', 'O'],
        'G': ['B', 'F', 'P', 'Q', 'R'],
        'H': ['C', 'G', 'S'],
        'I': ['D'],
        'J': ['D', 'T', 'U'],
        'K': ['E'],
        'L': ['E', 'V'],
        'M': ['F'],
        'N': ['F'],
        'O': ['F'],
        'P': ['G'],
        'Q': ['G'],
        'R': ['G'],
        'S': ['H', 'W', 'X'],
        'T': ['J'],
        'U': ['J', 'Y', 'Z'],
        'V': ['L'],
        'W': ['S'],
        'X': ['S'],
        'Y': ['U'],
        'Z': ['U']
}


# In[7]:


ids(graph2,'A','Y')


# In[8]:


bfs(graph2,'A','Y')


# In[ ]:




