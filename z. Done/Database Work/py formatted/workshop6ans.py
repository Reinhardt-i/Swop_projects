#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1

# In[1]:


temperatures = [-1.3, 0.1, 2.1, 2.0, 1.5, 0.3, 0.2, 0.8, 0.1, -0.1, -0.9, -1.0, 0.5]
filtered_temperatures = [temp for temp in temperatures if temp <= 0]
print(filtered_temperatures)


# ### Exercise 2

# In[2]:


stocks = {
    "UWMC": {"price": 6.98, "market_cap": 0.7, "pe_ratio": 1.2},
    "QRTEA": {"price": 10.36, "market_cap": 4.2, "pe_ratio": 3.1},
    "SAGE": {"price": 41.42, "market_cap": 2.4, "pe_ratio": 3.3},
    "NLY": {"price": 8.69, "market_cap": 12.6, "pe_ratio": 3.8},
    "SLVM": {"price": 27.64, "market_cap": 1.2, "pe_ratio": 4.2},
}


filtered_stocks = {
    symbol: info for symbol, info in stocks.items()
    if info["pe_ratio"] > 2 and info["market_cap"] > 5
}

print(filtered_stocks)


# ### Exercise 2 part 2

# In[4]:


num_cube_lc = [n**3 for n in range(1, 11) if n % 2 == 0]
num_cube_generator = (num**3 for num in range(1, 11) if num % 2 == 0)

list_comp_result = f"List Comprehension = {num_cube_lc}\n"
generator_exp_result = f"Generator Expression = {num_cube_generator}\n"

sum_result = f"Sum = {sum(num_cube_generator)}\n"
print(list_comp_result, generator_exp_result, sum_result)


# In[5]:


print([n**3 for n in range(1, 10*1000) if n % 2 == 0])


# ### Exercise 3

# In[ ]:


# Logged in, checked out gameUser


# ### Exercise 4

# #### Calculating Wasted Space -
# 
# 
# - The SQL query to get the maximum length of data stored in each varchar field :
# ```sql
# 
# SELECT 
#   MAX(LENGTH(id)) AS MaxIdLength,
#   MAX(LENGTH(phone)) AS MaxPhoneLength,
#   MAX(LENGTH(name)) AS MaxNameLength,
#   MAX(LENGTH(email)) AS MaxEmailLength,
#   MAX(LENGTH(address)) AS MaxAddressLength,
#   MAX(LENGTH(country)) AS MaxCountryLength,
#   MAX(LENGTH(region)) AS MaxRegionLength,
#   MAX(LENGTH(postalzip)) AS MaxPostalZipLength,
#   MAX(LENGTH(token)) AS MaxTokenLength
# FROM gameUser;
# 
# 
# ```
# 
# - The SQL query for the allocated space for each varchar field :
# ```sql
# 
# SELECT 
#   COLUMN_NAME, 
#   CHARACTER_MAXIMUM_LENGTH 
# FROM information_schema.COLUMNS 
# WHERE 
#   TABLE_NAME = 'gameUser' AND 
#   TABLE_SCHEMA = 'cetm50' AND
#   DATA_TYPE = 'varchar';
# 
# 
# ```
# 
# We can calculate the wasted space for each field like this:
# 
# 1. Calculate the wasted space for each field per record by subtracting the maximum used length from the allocated space.
# 2. Multiply the wasted space per record by the number of records to find the total wasted space for that field.
# 
# 
# The calculated wasted space for each varchar field in the database, as well as the total wasted space, is as follows (in bytes):
# 
# - Phone: 571,704 bytes
# - Name: 1,584,240 bytes
# - Email: 1,542,912 bytes
# - Address: 1,343,160 bytes
# - Country: 337,512 bytes
# - Region: 199,752 bytes
# - Postalzip: 6,888 bytes
# - Token: 1,708,224 bytes
# 
# **Total wasted space across all varchar fields for all records is 7,294,392 bytes, which is approximately 7,123.43 kilobytes (or about 6.96 megabytes).**

# ### Exercise 5

# *Checked out the links, and noted SQL things I didn't know*

# ### Exercise 6

# Here are the SQL queries :
# 
# 1. To select only the email addresses from the table:
# ```sql
# SELECT email FROM gameUser;
# ```
# (688 total, Query took 0.0006 seconds.)
# 
# 2. To select all users where their `gamerscore` exceeds 9000:
# ```sql
# SELECT * FROM gameUser WHERE gamerscore > 9000;
# ```
# (677 total, Query took 0.0012 seconds.)
# 
# 3. To select all users where their `gamerscore` exceeds 9000, ordering the results based on this `gamerscore`:
# ```sql
# SELECT * FROM gameUser WHERE gamerscore > 9000 ORDER BY gamerscore DESC;
# ```
# (677 total, Query took 0.0016 seconds.) [gamerscore: 80000... - 80000...]

# ### Exercise 7

# 
# ```sql
# 
# INSERT INTO gameUser (phone, name, email, address, country, region, postalzip, token, cash_spent, gamerscore)
# VALUES ('1234567890', 'Fake Me', 'fake.me@example.com', '123 Fake Street', 'Fakeland', 'Fakestate', 'FAKE123', 'tok1234567', 5000, 9500);
# 
# ```
# - OUTPUT :  1 row inserted. Inserted row id: 5255634 (Query took 0.0055 seconds.)
# 
# *Verifying -*
# 
# ```sql
# 
# SELECT * FROM gameUser WHERE id = 5255634;
# 
# ```
# OUTPUT : Showing rows 0 - 0 (1 total, Query took 0.0013 seconds.) 
# Full texts	
# id - 5255634, phone - 1234567890, name - Fake Me, email - fake.me@example.com, address - 123 Fake Street, country - Fakeland, region - Fakestate, postalzip - FAKE123, token - tok1234567, cash_spent - 5000, gamerscore - 9500

# ### Exercise 8

# #### Modifying my fabricated record -
# 
# ```sql
# 
# UPDATE gameUser 
# SET gamerscore = 12000, cash_spent = 3000, phone = '9876543210' 
# WHERE id = 5255634;
# 
# ```
#  1 row affected. (Query took 0.0033 seconds.)
# 
# Verifying : 
# 
# ```sql
# SELECT * FROM gameUser WHERE id = 5255634;
# ```
# id - 5255634, phone - 9876543210, name - Fake Me, email - fake.me@example.com, address - 123 Fake Street, country - Fakeland, region - Fakestate, postalzip - FAKE123, token - tok1234567, cash_spent - 3000, gamerscore - 12000

# ### Exercise 9

# Firstly, Dry-Run -
# 
# ```sql
# 
# START TRANSACTION;
# 
# DELETE FROM gameUser WHERE id = 5255634;
# 
# -- To check how many rows would be affected
# SELECT ROW_COUNT();
# 
# ROLLBACK;
# 
# 
# ```
# 
# OUTPUT :  1 row affected. (Query took 0.0022 seconds.)
# 
# So, I can actually run the query :
# 
# ```sql
# 
# DELETE FROM gameUser WHERE id = 5255634;
# 
# ```
# 
# OUTPUT : 1 row affected. (Query took 0.0033 seconds.)

# ### Exercise 10

# Installed pymysql and pony via PIP.
# 
# 

# ### Exercise 11

# In[11]:


from pony.orm import Database
db = Database()

# MySQL
db.bind(provider='mysql', host='europa.ashley.work', user='cetm50_user', passwd='iE93F2@8EhM@1zhD&u9M@K', db='cetm50')


# ### Exercise 12

# In[12]:


from pony.orm import Database
from pony.orm import db_session
db = Database()

# MySQL
db.bind(provider='mysql', host='europa.ashley.work', user='cetm50_user', passwd='iE93F2@8EhM@1zhD&u9M@K', db='cetm50')


with db_session:
    my_query_result = db.select("SELECT * FROM gameUser LIMIT 10;")
    

print(len(my_query_result))
print(my_query_result)


# ### Exercise 13

# In[15]:


from pony.orm import Database, db_session

db = Database()
db.bind(provider='mysql', host='europa.ashley.work', user='cetm50_user', passwd='iE93F2@8EhM@1zhD&u9M@K', db='cetm50')

# Query 1 :

with db_session:
    cursor = db.execute("""
        SELECT 
            MAX(LENGTH(id)) AS MaxIdLength,
            MAX(LENGTH(phone)) AS MaxPhoneLength,
            MAX(LENGTH(name)) AS MaxNameLength,
            MAX(LENGTH(email)) AS MaxEmailLength,
            MAX(LENGTH(address)) AS MaxAddressLength,
            MAX(LENGTH(country)) AS MaxCountryLength,
            MAX(LENGTH(region)) AS MaxRegionLength,
            MAX(LENGTH(postalzip)) AS MaxPostalZipLength,
            MAX(LENGTH(token)) AS MaxTokenLength
        FROM gameUser;
    """)
    max_lengths = cursor.fetchone()
    print(max_lengths)

    
    
# Query 2: Select Email Addresses

with db_session:
    emails = db.select("SELECT email FROM gameUser;")
    print(emails)

    
# Query 3: Select Users with Gamerscore Over 9000

with db_session:
    high_score_users = db.select("SELECT * FROM gameUser WHERE gamerscore > 9000;")
    print(high_score_users)

    
# Query 4: Order Users by Gamerscore
  
with db_session:
    ordered_high_score_users = db.select("SELECT * FROM gameUser WHERE gamerscore > 9000 ORDER BY gamerscore DESC;")
    print(ordered_high_score_users)

    
# Query 5: Insert Fabricated Data

user_data = {
    'phone': '1234567890',
    'name': 'Fake Me',
    'email': 'fake.me@example.com',
    'address': '123 Fake Street',
    'country': 'Fakeland',
    'region': 'Fakestate',
    'postalzip': 'FAKE123',
    'token': 'tok1234567',
    'cash_spent': 5000,
    'gamerscore': 9500
}


with db_session:
    db.execute(f"""
        INSERT INTO gameUser (phone, name, email, address, country, region, postalzip, token, cash_spent, gamerscore)
        VALUES ('{user_data['phone']}', '{user_data['name']}', '{user_data['email']}', '{user_data['address']}', 
        '{user_data['country']}', '{user_data['region']}', '{user_data['postalzip']}', '{user_data['token']}', 
        {user_data['cash_spent']}, {user_data['gamerscore']});
    """)


    
# Query 6: Select User by ID

with db_session:
    user_by_id = db.select("SELECT * FROM gameUser WHERE id = 5255634;")
    print(user_by_id)

    
# Query 7: Update User Record

with db_session:
    db.execute("UPDATE gameUser SET gamerscore = 12000, cash_spent = 3000, phone = '9876543210' WHERE id = 5255634;")

    
# Query 8: Select Updated User by ID

with db_session:
    updated_user_by_id = db.select("SELECT * FROM gameUser WHERE id = 5255634;")
    print(updated_user_by_id)

    
# Query 9: Delete User Record

with db_session:
    db.execute("DELETE FROM gameUser WHERE id = 5255634;")

