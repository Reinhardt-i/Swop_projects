#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1

# In[1]:


def my_amazing_func():
       print("Hello World! From my function!")


# In[2]:


my_amazing_func()


# ### Exercise 2

# In[3]:


variable_name = my_amazing_func()
print(variable_name)
print(type(variable_name))


# ### Exercise 3

# In[4]:


def my_amazing_func( thing_to_print ):
       print( thing_to_print )
        
my_amazing_func("test string!")


# ### Exercise 4

# In[7]:


def my_amazing_func(thing_to_print = "The Default thing you want"):
      print( thing_to_print )
        
my_amazing_func()
my_amazing_func("test string!")


# ### Exercise 5

# In[12]:


import random

x = random.randint(0, 10)
print(x)


# ### Exercise 6

# In[14]:


for i in range(20):
    print(random.randint(0, 100), end=" ")


# ### Exercise 7

# In[15]:


a = "Some outer scope string literal"
b = 42


def some_func():
    print(a, b)

    
def some_func_v2(c, d):
    print(c, d)
    

some_func()
# some_func_v2()  # This line would error "positional arguments"
some_func_v2( a, b )
    


# ### Exercise 8

# In[16]:


def always_4():
       return 4
    
four = always_4()
print(four)


# ### Exercise 9

# In[17]:


def addition(a, b):
    return a + b

print(addition(6, 4))


# ### Exercise 10

# In[18]:


def find_negative_one(A):
    for i in range(len(A)):
        if A[i] == -1:
            return i
    return None


A = [ 5, 2, 9, -1, 3, 12 ]
indx_of_issue = find_negative_one( A )
print(indx_of_issue)


# ### Exercise 11

# In[19]:


def find_negative_one(A):
    for i in range(len(A)):
        if A[i] == -1:
            return i
    return "There is no -1 here"


A = [ 5, 2, 9, 3, 12 ]
indx_of_issue = find_negative_one( A )
print(indx_of_issue)


# ### Exercise 12

# In[30]:


def get_combined_namegrades(names, grades):

    records = {}

    if len(names) == len(grades):
        for i in range(len(names)):
            records[names[i]] = grades[i]
    else:
        print("Error: The lists of names and grades are not of the same length.")

    return records


student_records = {
   "Ada": 98.0,
   "Bill": 45.0,
   "Charlie": 63.2
}


student_names = ["Teri", "Johanna", "Tomas", "Piotr", "Grzegorz"]
student_grades = [35.0, 52.5, 37.8, 65.0, 64.8]


new_records = get_combined_namegrades(student_names, student_grades)
student_records.update(new_records)
print(student_records)


# ### Exercise 13

# In[32]:


filtered_student_records = {student: grade for student, grade in student_records.items() if grade >= 65}
print(filtered_student_records)


# ### Exercise 14

# In[36]:


def grade_to_classification(grade):
    if grade < 40.0:
        return "Fail"
    elif grade < 50.0:
        return "Pass"
    elif grade < 60.0:
        return "2:2"
    elif grade < 70.0:
        return "2:1"
    else:
        return "First"


classified_student_records = {student: grade_to_classification(grade) for student, grade in student_records.items()}
print(classified_student_records)


# ### Exercise 15

# In[38]:


more_grades = [0.0, 50.0, 49.9, 79.0, 101.0, 65.0, 54.2, 48.2, 78.9]
classified_grades = [grade_to_classification(grade) for grade in more_grades]
print(classified_grades)


# ### Exercise 16

# In[39]:


failed_grades = [grade_to_classification(grade) for grade in more_grades if grade < 40]
number_of_failures = len(failed_grades)
print(number_of_failures)


# ### Exercise 17

# In[41]:


adjusted_grades = [grade if grade <= 100 else 100 for grade in more_grades]
print(adjusted_grades)


# ### Exercise 18

# In[51]:


class Student(object):
    pass

alex = Student()


# ### Exercise 19

# In[52]:


print(type(alex))
print(isinstance(alex, Student))


# ### Exercise 20

# In[53]:


class Student(object):
    def __init__(self):
        print("This gets called when I make a new student.")


alex = Student()


# ### Exercise 21

# In[55]:


class Student(object):
    def __init__(self, name, grade):
        self.name = name
        self.grade = float(grade)


# alex = Student()
# The line above induces TypeError: Student.__init__() missing 2 required positional arguments: 'name' and 'grade'

alex = Student("Alex", 99)


# ### Exercise 22

# In[58]:


class Student(object):
    def __init__(self, name, grade):
        print("This gets called when I make a new student.")
        print(f"Creating student {name}, with grade {grade}")
        self.name = name
        self.grade = float(grade)

alex = Student("Alex", 99)
some_students = [Student("Alex", 99), Student("Rob", 35.0), Student("Tasha", 70.0)]


# ### Exercise 23

# In[61]:


class Student(object):
    def __init__(self, name, grade):
        self.name = name
        self.grade = float(grade)

    def get_classification(self):
        if self.grade < 40.0:
            return "Fail"
        elif self.grade < 50.0:
            return "Pass"
        elif self.grade < 60.0:
            return "2:2"
        elif self.grade < 70.0:
            return "2:1"
        else:
            return "First"


alex = Student("Alex", 99)
some_students = [Student("Alex", 99), Student("Rob", 35.0), Student("Tasha", 70.0)]
print(some_students[0], end="\n\n")


print(alex.get_classification())


# ### Extended Exercise 1

# In[62]:


def is_palindrome(input_list):
    return [s.replace(" ", "").lower() == s.replace(" ", "").lower()[::-1] for s in input_list]


input_list = ["taco cat", "bob", "davey"]
output = is_palindrome(input_list)
print(output)


# In[ ]:




