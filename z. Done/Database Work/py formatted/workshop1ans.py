#!/usr/bin/env python
# coding: utf-8

# # CETM50 - Workshop 1
# ## Student Name: Abraar
# ## Student ID: <StudentID>

# ### Exercise 1

# In[2]:


print("Hello World!")


# ### Exercise 2

# In[9]:


print(type(932))
print(type(10.0))
print(type("Hello!"))
print(type([10, 12, 15, 20, 21]))
print(type((10, 12, 15)))
print(type(10 == 11))


# ### Exercise 3

# In[10]:


type(str(932))


# ### Exercise 4

# In[13]:


print(type(3+4))
print(type(3.0+4))
print(type(37%7**2))
print(type('bob' + 'cat'))
# Next line is commented out because it induces a TypeError: unsupported operand type(s) for /: 'str' and 'str'
# print(type("bob" / "cat"))  
print(type("banana" + "na" * 20))


# ### Exercise 5 
# 
# Acording to PEMDAS, 
# 
# First, we evaluate the expression inside parentheses or brackets. However, there are no parentheses in this expression. Next, we move to Exponents/Orders, but there are no exponents here either.
# 
# Then, we perform Multiplication and Division from left to right:
# 28 / 7 = 4, then,
# 4 * 2 = 8
# Finally, we perform Addition:
# 12 + 8 = 20
# So, the result of the expression "12 + 28 / 7 * 2" is 20.
# 
# The overall type of the result is an integer.
# 
# To make the expression less ambiguous, we can use parentheses to explicitly specify the order of operations.
# Here's a way -
# 12 + ((28 / 7) * 2)

# In[17]:


x = 12 + 28 / 7 * 2
x_less_ambiguous = 12 + ((28 / 7) * 2)
print(x)
print(x_less_ambiguous)
print(type(x))
print(type(x_less_ambiguous))


# ### Exercise 6

# In[18]:


# Define the variables
a = 42.0
b = 97

# Perform calculations
addition = a + b
subtraction = a - b
division = a / b
multiplication = a * b

# Print the results
print(f"Addition: {addition}")
print(f"Subtraction: {subtraction}")
print(f"Division: {division}")
print(f"Multiplication: {multiplication}")


# ### Exercise 7

# In[19]:


print(multiplication)
multiplication = "banana" + "na" * 20
print(multiplication)


# ### Exercise 10

# In[22]:


result1 = "ada" < "bill"
print(f'Comparison "ada" < "bill" is {result1}')


result2 = "ada" < "adb"
print(f'Comparison "ada" < "adb" is {result2}')


result3 = "ada" < "adalovelace"
print(f'Comparison "ada" < "adalovelace" is {result3}')


# ### Exercise 11

# In[23]:


user_name = input("Please enter your name: ")
my_name = "Abraar"

if user_name == my_name:
    print("Hello, it's me!")
else:
    print("Hello, someone else!")


# ### Exercise 12

# In[33]:


user_name = input("Please enter your name: ")

my_first_name = "Abraar"
my_last_name = "Nafiz"

if user_name == my_first_name or user_name == my_last_name:
    print("Hello, it's me!")
else:
    print("Hello, someone else!")


# ### Exercise 13 :
# | a | b | c | a and b | (a and b) or c |
# |---|---|---|---------|----------------|
# | True | True | True | True | True |
# | True | False | True | False | True |
# | False | True | True | False | True |
# | False | False | True | False | True |
# | True | True | False | True | True |
# | True | False | False | False | False |
# | False | True | False | False | False |
# | False | False | False | False | False |
# 

# ### Exercise 14

# | a | b | c | a and b | (a and b) or c | (a and b) or not c |
# |---|---|---|---------|----------------|--------------------|
# | True | True | True | True | True | True |
# | True | False | True | False | True | False |
# | False | True | True | False | True | False |
# | False | False | True | False | True | False |
# | True | True | False | True | True | True |
# | True | False | False | False | False | True |
# | False | True | False | False | False | True |
# | False | False | False | False | False | True |
# 

# ### Exercise 15

# In[31]:


def is_even(x : int):
    if x % 2 == 0:
        print(f"The variable {x} is even!")


is_even(10)
is_even(9)
is_even(103)
is_even(0)
is_even(44)


# ### Exercise 16

# In[32]:


def check_parity(x : int):
    if x % 2 == 0:
        print(f"The variable {x} is even!")
    else:
        print(f"The variable {x} is odd!")
        
check_parity(10)
check_parity(9)
check_parity(103)
check_parity(0)
check_parity(44)


# ### Exercise 17

# In[35]:


def check_divisibility(number):
    if number % 2 == 0 and number % 3 == 0:
        print(f"{number} is divisible by both 2 and 3")
    elif number % 2 == 0:
        print(f"{number} is divisible by 2")
    elif number % 3 == 0:
        print(f"{number} is divisible by 3")
    else:
        print(f"{number} is not divisible by 2 or 3")


check_divisibility(6)
check_divisibility(17)
check_divisibility(9)
check_divisibility(4)
check_divisibility(60)


# ### Exercise 18

# In[42]:


A = [5, 2, 9, -1, 3, 12]

counter = 0
while counter < len(A):
    item = A[counter]
    print(f"Item: {item}", end=", ")
    
    if item == -1:
        print("\nEncountered -1, breaking out of the loop!")
        break

    square = item ** 2
    print(f"[Square of {item}] = {square}")
    
    counter += 1


# ### Exercise 19

# In[43]:


A = [5, 2, 9, -1, 3, 12]

for i in range(len(A)):
    item = A[i]
    print(f"Item: {item}", end=", ")

    if item == -1:
        print("\nEncountered -1, breaking out of the loop!")
        break

    square = item ** 2
    print(f"[Square of {item}] = {square}")


# ### Exercise 20

# In[44]:


A = [5, 2, 9, -1, 3, 12]

for item in A:
    print(f"Item: {item}", end=", ")

    if item == -1:
        print("\nEncountered -1, breaking out of the loop!")
        break

    square = item ** 2
    print(f"[Square of {item}] = {square}")


# ### Exercise 21

# In[49]:


A = [5, 2, 9, -1, 3, 12]

sum_of_numbers = 0

for item in A:
    sum_of_numbers += item

print(f"The sum of all numbers in the list {A} is  ->  {sum_of_numbers}")


# ### Exercise 22

# In[50]:


A = [5, 2, 9, -1, 3, 12]

mean = sum(A) / len(A)

print(f"The mean of the numbers in the list {A} is  ->  {mean}")


# ### Exercise 23

# In[29]:


my_items = [ -5, 3, 72, 1, 9, 24, -3]
max_so_far = None

for elem in my_items:
    if max_so_far == None or elem > max_so_far:
        max_so_far = elem

print(f"Maximum Value: {max_so_far}")


# ### Exercise 24

# In[51]:


A = 5
B = A
B = 10
print(A)  # Output: 5
print(B)  # Output: 10


A = [1, 6, 2, 7]
B = A
B.remove(6)
print(A)  # Output: [1, 2, 7]


A = [1, 6, 2, 7]
B = A[:]
B.remove(6)
print(A)  # Output: [1, 6, 2, 7]
print(B)  # Output: [1, 2, 7]


# ### Exercise 25

# In[53]:


A = [1, 6, 2, 7]
B = A[:]

B.append(7)
B.append(42)
B.append(100)
B.append(370899)

print("List A:", A)  # This will print the original list A
print("List B:", B)  # This will print the modified list B


# ### Exercise 26

# In[56]:


bought_cost = [10.0, 12.55, 17.99]
sale_price = [12.0, 11.50, 20.0]

results = []
total_profit = 0

for cost, sale in zip(bought_cost, sale_price):
    profit = round(sale - cost, 4)  # Rounded to 4 decimal places
    results.append(profit)
    total_profit += profit

total_profit = round(total_profit, 4)  # Round total profit to 4 decimal places

print(f"Profit/Loss per item : {results}")

if total_profit > 0:
    print(f"Total Profit : {total_profit}")
else:
    print(f"Total Loss : {abs(total_profit)}")


# ### Exercise 26(b)

# In[57]:


student_records = {
    "Ada": 98.0,
    "Bill": 45.0,
    "Charlie": 63.2
}

student_names = ["Neva", "Kelley", "Emerson"]
student_grades = [72.2, 64.9, 32.0]

if len(student_names) == len(student_grades):
    for i in range(len(student_names)):
        student_records[student_names[i]] = student_grades[i]
else:
    print("Error: The lists of names and grades are not of the same length.")

print(student_records)


# ### Exercise 27

# In[59]:


for k, v in student_records.items():
    print(f"Student: {k},   Grade: {v}")


# ### Extended Exercise 1

# In[61]:


def bubble_sort(L):
    
    n = len(L)
    for i in range(n):
        
        swapped = False
        # Last i elements are already in place
        for j in range(0, n-i-1):
            if L[j] > L[j+1]:
                L[j], L[j+1] = L[j+1], L[j]
                swapped = True
                
        if not swapped:
            break
            
    return L


L = [9, 2, 12, 7]
sorted_L = bubble_sort(L)
print(sorted_L)


# In[ ]:




