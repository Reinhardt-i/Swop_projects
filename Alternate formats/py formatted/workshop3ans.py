#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1

# In[1]:


f = open("ex1_data.txt", "x")
f.close()


# In[2]:


f = open("ex1_data.txt", "a")
f.write("I got my first real six-string\nBought it at the five and dime\nPlayed it 'til my fingers bled \nWas the summer of '69 \nMe and some guys from school\nHad a band and we tried real hard\nJimmy quit and Jody got married\nI should've known we'd never get far.")
f.close()


# ### Exercise 2

# In[3]:


my_file = open('ex1_data.txt', mode='r')
print(my_file)
print(type(my_file))


# ### Exercise 3

# In[6]:


with open('ex1_data.txt', 'r') as file:
    for line in file:
        print(line)


# ### Exercise 4

# In[7]:


my_file.close()


# ### Exercise 5

# In[8]:


word = "it"

with open('ex1_data.txt', 'r') as file:
    for line in file:
        if word in line:
            print("Found!")


# ### Exercise 6

# In[14]:


word = "it"

with open('ex1_data.txt', 'r') as file:
    for line_no, line in enumerate(file):
        if word in line.lower():
            print(f"Found! At: {line_no}")
            


# ### Exercise 7

# In[15]:


word = "it"
lines_with_word = []

with open('ex1_data.txt', 'r') as file:
    for line_no, line in enumerate(file):
        if word in line.lower():
            print(f"Found! At: {line_no}")
            lines_with_word.append(line.rstrip())

for line in lines_with_word:
    print(line)


# ### Exercise 8

# In[17]:


ex8_file = open('ex1_data_copy.txt', mode='w')
print( ex8_file )
print( type(ex8_file) )


# ### Exercise 9

# In[18]:


content = "We're taking the hobbits to the Isengard\n"

with open('ex1_data_copy.txt', 'w') as file:
    file.write(content)


# ### Exercise 10

# In[19]:


lines_to_append = ["Doe, a deer\n", "a female deer\n", "far\n", "a long long way to run!\n"]

with open('ex1_data_copy.txt', 'a') as file:
    file.writelines(lines_to_append)


# ### Exercise 11

# In[ ]:


# Already did prev ones with context managers!


# ### Exercise 12

# In[ ]:


try:
    with open("no_exist.txt", 'r') as file:
        pass
except FileNotFoundError:
    print("FileNotFoundError: The file does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")


# ### Exercise 13

# In[21]:


try:
    with open("no_exist.txt", 'r') as file:
        pass
except:
    print(f"An error occurred")


# ### Exercise 14

# In[22]:


try:
    with open("no_exist.txt", 'r') as file:
        pass
except FileNotFoundError:
    print("FileNotFoundError: The file does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")


# ### Exercise 15

# In[ ]:


# Downloaded!


# ### Exercise 16

# In[29]:


import csv

file_path = 'wind_data.csv'

with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    counter = 0
    for line in csv_reader:
        if counter > 200:
            break
        print(line)
        counter += 1


# ### Exercise 17

# In[30]:


with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)

    print("Headers:", headers)
    print("Type of headers:", type(headers))


# ### Exercise 18

# In[33]:


wind_speeds = []
with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    wind_speed_index = headers.index('Wind Speed (m/s)')
    for line in csv_reader:
        wind_speed = float(line[wind_speed_index])
        wind_speeds.append(wind_speed)

print(wind_speeds)


# ### Exercise 19

# In[34]:


num_wind_speed_records = len(wind_speeds)
average_wind_speed = sum(wind_speeds) / num_wind_speed_records
min_wind_speed = min(wind_speeds)
max_wind_speed = max(wind_speeds)

print("Number of records:", num_wind_speed_records)
print("Average wind speed:", average_wind_speed)
print("Minimum wind speed:", min_wind_speed)
print("Maximum wind speed:", max_wind_speed)


# ### Exercise 20

# In[35]:


output_file_path = 'wind_speeds.csv'

with open(output_file_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    for wind_speed in wind_speeds:
        csv_writer.writerow([wind_speed])

print("Wind speeds written to:", output_file_path)


# ### Exercise 21

# In[36]:


import numpy as np

x = np.array(wind_speeds)
print("Mean:", x.mean())
print("Max:", x.max())
print("Min:", x.min())
print("Standard Deviation:", x.std())
print("Variance:", x.var())


# ### Exercise 22

# In[37]:


import json

duck_1 = {"first_name": "Davey", "last_name": "McDuck", "followers": 12865, "following": 120}
duck_2 = {"first_name": "Jim", "last_name": "Bob", "followers": 123, "following": 5000}
duck_3 = {"first_name": "Celest", "last_name": "", "followers": 40189, "following": 1}

duck_collection = [duck_1, duck_2, duck_3]

print(duck_collection)


# ### Exercise 23

# In[38]:


import json

duck_1 = {"first_name": "Davey", "last_name": "McDuck", "followers": 12865, "following": 120}
duck_2 = {"first_name": "Jim", "last_name": "Bob", "followers": 123, "following": 5000}
duck_3 = {"first_name": "Celest", "last_name": "", "followers": 40189, "following": 1}

duck_collection = [duck_1, duck_2, duck_3]

with open('ducks.json', 'w') as file:
    json.dump(duck_collection, file)


# ### Exercise 24

# In[39]:


import json

duck_1 = {"first_name": "Davey", "last_name": "McDuck", "followers": 12865, "following": 120}
duck_2 = {"first_name": "Jim", "last_name": "Bob", "followers": 123, "following": 5000}
duck_3 = {"first_name": "Celest", "last_name": "", "followers": 40189, "following": 1}

duck_collection = [duck_1, duck_2, duck_3]

with open('ducks.json', 'w') as file:
    json.dump(duck_collection, file)

with open('ducks.json', 'r') as file:
    loaded_ducks = json.load(file)

print(duck_collection == loaded_ducks)


# ### Exercise 25

# In[40]:


import json

duck_1 = {"first_name": "Davey", "last_name": "McDuck", "followers": 12865, "following": 120}
duck_2 = {"first_name": "Jim", "last_name": "Bob", "followers": 123, "following": 5000}
duck_3 = {"first_name": "Celest", "last_name": "", "followers": 40189, "following": 1}

duck_collection = [duck_1, duck_2, duck_3]

with open('ducks.json', 'w') as file:
    json.dump(duck_collection, file)

with open('ducks.json', 'r') as file:
    loaded_ducks = json.load(file)

trendy_ducks = []
for duck in loaded_ducks:
    net_followers = duck["followers"] - duck["following"]
    trendy_ducks.append(net_followers)
    print(f"{duck['first_name']} net followers: {net_followers}")


# ### Exercise 26

# In[42]:


import json


duck_1 = {"first_name": "Davey", "last_name": "McDuck", "followers": 12865, "following": 120}
duck_2 = {"first_name": "Jim", "last_name": "Bob", "followers": 123, "following": 5000}
duck_3 = {"first_name": "Celest", "last_name": "", "followers": 40189, "following": 1}

duck_collection = [duck_1, duck_2, duck_3]


with open('ducks.json', 'w') as file:
    json.dump(duck_collection, file)


with open('ducks.json', 'r') as file:
    loaded_ducks = json.load(file)

trendy_ducks = []
for duck in loaded_ducks:
    net_followers = duck["followers"] - duck["following"]
    trendy_ducks.append(net_followers)
    # print(f"{duck['first_name']} net followers: {net_followers}")

    
arr_trendy_ducks = np.array(trendy_ducks)
trendiest_duck_index = arr_trendy_ducks.argmax()
print("Trendiest duck:", duck_collection[trendiest_duck_index]["first_name"], "Net followers:", arr_trendy_ducks[trendiest_duck_index])


# ### Exercise 27

# In[44]:


positive_net_followers_indices = np.where(arr_trendy_ducks > 0)[0]
positive_net_followers_ducks = [duck_collection[i] for i in positive_net_followers_indices]

print(len(positive_net_followers_ducks))
print(positive_net_followers_ducks)


# ### Exercise 28

# In[47]:


positive_net_followers_indices = np.where(arr_trendy_ducks > 0)[0]
positive_net_followers_ducks = [duck_collection[i] for i in positive_net_followers_indices]

print(arr_trendy_ducks % 2 == 0)
print(np.where(arr_trendy_ducks % 2 == 0))

actual_indices = np.where(arr_trendy_ducks % 2 == 0)[0]
print(actual_indices)
print(type(actual_indices))
print(type(list(actual_indices)))


# ### Exercise 29

# In[48]:


positive_net_followers_indices = np.where(arr_trendy_ducks > 0)[0]
positive_net_followers_ducks = [duck_collection[i] for i in positive_net_followers_indices]

with open('positive_net_followers_ducks.json', 'w') as file:
    json.dump(positive_net_followers_ducks, file)


# In[ ]:





# In[ ]:




