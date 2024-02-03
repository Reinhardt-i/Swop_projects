#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1

# In[39]:


from pony.orm import *
import pymysql.cursors
from decimal import Decimal
from datetime import date
from pony.orm import Database, db_session
from pony.orm import PrimaryKey, Required
from pony.orm import set_sql_debug


db = Database()


class Cat(db.Entity):
    id = PrimaryKey(int, auto=True)
    name = Required(str, 255)
    age = Required(int)
    breed = Required(str, 255)
    color = Required(str, 255)
    medical_history = Optional(LongStr)
    current_medications = Optional(LongStr)
    owner_name = Required(str, 255)
    owner_contact = Required(str, 255)
    

class Agent(db.Entity):
    _table_ = "AGENTS"
    agent_code = PrimaryKey(str, 6)
    agent_name = Optional(str, 40)
    working_area = Optional(str, 35)
    commission = Optional(Decimal, precision=10, scale=2)
    phone_no = Optional(str, 15)
    country = Optional(str, 25)
    orders = Set('Order')  # Relationship to ORDERS
    customers = Set('Customer')  # Relationship to CUSTOMER

    
class Customer(db.Entity):
    _table_ = "CUSTOMER"
    cust_code = PrimaryKey(str, 6)
    cust_name = Required(str, 40)
    cust_city = Optional(str, 35)
    working_area = Required(str, 35)
    cust_country = Required(str, 20)
    grade = Optional(float)
    opening_amt = Required(Decimal, precision=12, scale=2)
    receive_amt = Required(Decimal, precision=12, scale=2)
    payment_amt = Required(Decimal, precision=12, scale=2)
    outstanding_amt = Required(Decimal, precision=12, scale=2)
    phone_no = Optional(str)
    agent_code = Optional(Agent, column='AGENT_CODE')
    orders = Set('Order')

    
class Order(db.Entity):
    _table_ = "ORDERS"
    ord_num = PrimaryKey(int)
    ord_amount = Required(Decimal, precision=12, scale=2)
    advance_amount = Required(Decimal, precision=12, scale=2)
    ord_date = Required(date)
    cust_code = Required('Customer', column='CUST_CODE')
    agent_code = Required('Agent', column='AGENT_CODE')
    ord_description = Required(str, 60)    

    
db.bind(
    provider='mysql',
    host='europa.ashley.work',
    user='student_bi56is',
    passwd='iE93F2@8EhM@1zhD&u9M@K',
    db='student_bi56is'
)


set_sql_debug(True)
db.generate_mapping(create_tables=False)
# db.generate_mapping(create_tables=True, check_tables=True)


# ### Exercise 2

# In[44]:


from statistics import mode


# 1. Select ALL Customers
@db_session
def select_all_customers():
    return select(c for c in Customer)[:]

customers = select_all_customers()
count = 0
for customer in customers:
    if count > 10:
        break
    else:
        print(customer)
        count += 1
print("\n----------------------------------------------------------------------\n")


# 2. Select All Orders & Sum Order Amounts
@db_session
def sum_all_order_amounts():
    order_amounts = select(o.ord_amount for o in Order)[:]
    return sum(order_amounts)

total_order_amount = sum_all_order_amounts()
print("\n----------------------------------------------------------------------\n")
print(f"Total Order Amount: {total_order_amount}")
print("\n----------------------------------------------------------------------\n")



# 3. Obtain the MAX Commission of the Agents
@db_session
def max_agent_commission():
    return max(a.commission for a in Agent)

max_commission = max_agent_commission()
print("\n----------------------------------------------------------------------\n")
print(f"Max Agent Commission: {max_commission}")
print("\n----------------------------------------------------------------------\n")



# 4. Obtain the MODE of the Working Area of the Agents
@db_session
def get_all_working_areas():
    return select(a.working_area for a in Agent)[:]


@db_session
def mode_working_area():
    working_areas = get_all_working_areas()
    working_area_list = list(working_areas)
    return mode(working_area_list)


working_area_mode = mode_working_area()
print("\n----------------------------------------------------------------------\n")
print(f"Mode of Working Area: {working_area_mode}")
print("\n----------------------------------------------------------------------\n")



# 5. Create a New Customer
@db_session
def create_customer(cust_code, cust_name, cust_city, working_area, cust_country, grade, opening_amt, receive_amt, payment_amt, outstanding_amt, phone_no, agent_code):
    if Customer.get(cust_code=cust_code):
        raise ValueError(f"A customer with cust_code {cust_code} already exists.")
    
    agent = Agent.get(agent_code=agent_code)
    if not agent:
        raise ValueError("Agent with the specified code does not exist.")
    
    Customer(
        cust_code=cust_code, cust_name=cust_name, cust_city=cust_city,
        working_area=working_area, cust_country=cust_country, grade=grade,
        opening_amt=opening_amt, receive_amt=receive_amt, payment_amt=payment_amt,
        outstanding_amt=outstanding_amt, phone_no=phone_no, agent_code=agent
    )

try:
    create_customer(
        cust_code="C12351",  # Make sure this is a new, unique code
        cust_name="New Customer", cust_city="Some City",
        working_area="Some Area", cust_country="Some Country", grade=2,
        opening_amt=Decimal("5000.00"), receive_amt=Decimal("1000.00"),
        payment_amt=Decimal("500.00"), outstanding_amt=Decimal("4500.00"),
        phone_no="1234567890", agent_code="A001"
    )
    print("\n----------------------------------------------------------------------\n")
    print("Customer created successfully.")
except ValueError as e:
    print(e)
    
print("\n----------------------------------------------------------------------\n")
print("\n----------------------------------------------------------------------\n")


# ### Exercise 3

# - Created Cat Table on the top, here's a screenshot -
# ![CatTable](https://github.com/Reinhardt-i/Random-Codes/raw/main/RandomData/CatTable.png)

# ### Exercise 4
# 
# - added set_sql_debug(True), outputs :
# 
# ```
# GET CONNECTION FROM THE LOCAL POOL
# SELECT `AGENTS`.`agent_code`, `AGENTS`.`agent_name`, `AGENTS`.`working_area`, `AGENTS`.`commission`, `AGENTS`.`phone_no`, `AGENTS`.`country`
# FROM `AGENTS` `AGENTS`
# WHERE 0 = 1
# 
# SELECT `CUSTOMER`.`cust_code`, `CUSTOMER`.`cust_name`, `CUSTOMER`.`cust_city`, `CUSTOMER`.`working_area`, `CUSTOMER`.`cust_country`, `CUSTOMER`.`grade`, `CUSTOMER`.`opening_amt`, `CUSTOMER`.`receive_amt`, `CUSTOMER`.`payment_amt`, `CUSTOMER`.`outstanding_amt`, `CUSTOMER`.`phone_no`, `CUSTOMER`.`AGENT_CODE`
# FROM `CUSTOMER` `CUSTOMER`
# WHERE 0 = 1
# 
# SELECT `ORDERS`.`ord_num`, `ORDERS`.`ord_amount`, `ORDERS`.`advance_amount`, `ORDERS`.`ord_date`, `ORDERS`.`CUST_CODE`, `ORDERS`.`AGENT_CODE`, `ORDERS`.`ord_description`
# FROM `ORDERS` `ORDERS`
# WHERE 0 = 1
# 
# SELECT `cat`.`name`, `cat`.`age`
# FROM `cat` `cat`
# WHERE 0 = 1
# 
# RELEASE CONNECTION
# 
# ```
# 

# ### Exercise 5

# In[34]:


"""
with db_session:
       my_cat = Cat(name="Kira", age=2)
"""

# Outputted :
"""
GET CONNECTION FROM THE LOCAL POOL
INSERT INTO `cat` (`name`, `age`) VALUES (%s, %s)
['Kira', 2]

COMMIT
RELEASE CONNECTION
"""

# Don't run this anymore plese, cat table has been updated with some required values.


# ### Exercise 6

# Firstly, I needed to change the cat table, so I did this:
# 
# ```sql
# 
# ALTER TABLE cat DROP PRIMARY KEY;
# ALTER TABLE cat ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY;
# ALTER TABLE cat ADD COLUMN breed VARCHAR(255) NOT NULL;
# ALTER TABLE cat ADD COLUMN color VARCHAR(255) NOT NULL;
# ALTER TABLE cat ADD COLUMN medical_history TEXT;
# ALTER TABLE cat ADD COLUMN current_medications TEXT;
# ALTER TABLE cat ADD COLUMN owner_name VARCHAR(255) NOT NULL;
# ALTER TABLE cat ADD COLUMN owner_contact VARCHAR(255) NOT NULL;
# 
# ```
# 
# 
# OUTPUT :
#  MySQL returned an empty result set (i.e. zero rows). (Query took 0.0151 seconds.)
# ALTER TABLE cat DROP PRIMARY KEY
# [Edit inline] [ Edit ] [ Create PHP code ]
#  MySQL returned an empty result set (i.e. zero rows). (Query took 0.0086 seconds.)
# ALTER TABLE cat ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY
# [Edit inline] [ Edit ] [ Create PHP code ]
#  MySQL returned an empty result set (i.e. zero rows). (Query took 0.0052 seconds.)
# ALTER TABLE cat ADD COLUMN breed VARCHAR(255) NOT NULL
# [Edit inline] [ Edit ] [ Create PHP code ]
#  MySQL returned an empty result set (i.e. zero rows). (Query took 0.0042 seconds.)
# ALTER TABLE cat ADD COLUMN color VARCHAR(255) NOT NULL
# [Edit inline] [ Edit ] [ Create PHP code ]
#  MySQL returned an empty result set (i.e. zero rows). (Query took 0.0047 seconds.)
# ALTER TABLE cat ADD COLUMN medical_history TEXT
# [Edit inline] [ Edit ] [ Create PHP code ]
#  MySQL returned an empty result set (i.e. zero rows). (Query took 0.0046 seconds.)
# ALTER TABLE cat ADD COLUMN current_medications TEXT
# [Edit inline] [ Edit ] [ Create PHP code ]
#  MySQL returned an empty result set (i.e. zero rows). (Query took 0.0050 seconds.)
# ALTER TABLE cat ADD COLUMN owner_name VARCHAR(255) NOT NULL
# [Edit inline] [ Edit ] [ Create PHP code ]
#  MySQL returned an empty result set (i.e. zero rows). (Query took 0.0043 seconds.)
# ALTER TABLE cat ADD COLUMN owner_contact VARCHAR(255) NOT NULL

# 
# Then, I changed the Cat class to this -
# ```python
# 
# class Cat(db.Entity):
#     id = PrimaryKey(int, auto=True)
#     name = Required(str, 255)
#     age = Required(int)
#     breed = Required(str, 255)
#     color = Required(str, 255)
#     medical_history = Optional(LongStr)
#     current_medications = Optional(LongStr)
#     owner_name = Required(str, 255)
#     owner_contact = Required(str, 255)
# 
# ```
# 

# ### Exercise 7

# In[46]:


with db_session:
    cats = [
        Cat(name="Kira", age=2, breed="Siamese", color="White", owner_name="Alice", owner_contact="123-456-7890"),
        Cat(name="Leo", age=3, breed="Bengal", color="Golden", owner_name="Bob", owner_contact="987-654-3210"),
        Cat(name="Max", age=1, breed="British Shorthair", color="Gray", owner_name="Charlie", owner_contact="555-666-7777"),
        Cat(name="Luna", age=4, breed="Persian", color="Black", medical_history="Healthy", owner_name="David", owner_contact="222-333-4444"),
        Cat(name="Bella", age=5, breed="Maine Coon", color="Brown", current_medications="Vitamins", owner_name="Eve", owner_contact="111-222-3333"),
        Cat(name="Oliver", age=2, breed="Ragdoll", color="Blue", owner_name="Fiona", owner_contact="666-777-8888"),
        Cat(name="Milo", age=1, breed="Sphynx", color="Pink", owner_name="George", owner_contact="999-888-7777")
    ]

        
"""
If you try to create a Cat object without specifying a required field 
(other than the auto-incrementing primary key), PonyORM will raise an exception, 
indicating that a required field is missing.
"""

with db_session:
    try:
        # Attempt to create a Cat without a required field (e.g., 'name')
        incomplete_cat = Cat(age=2, breed="Unknown", color="Grey", owner_name="Emily", owner_contact="444-555-6666")
    except Exception as e:
        print(f"Error: {e}")


# ### Exercise 8

# In[48]:


with db_session:
    kira_cats = Cat.select(lambda c: c.name == "Kira")[:]
    if kira_cats:
        for cat in kira_cats:
            print(f"Cat ID: {cat.id}, Name: {cat.name}, Age: {cat.age}")
    else:
        print("No cat named Kira found.")


# ### Exercise 9

# In[51]:


with db_session:
    all_cats = select(c for c in Cat)
    print("\n----------------------------------------------------------------------\n")
    print("Total number of cats:", all_cats.count())
    print("\n----------------------------------------------------------------------")
    print("----------------------------------------------------------------------\n")
    for cat in all_cats:
        print(f"Name: {cat.name}, Age: {cat.age}")
    print("\n----------------------------------------------------------------------")
    print("----------------------------------------------------------------------\n")


# ### Exercise 10

# In[52]:


from pony.orm import select, db_session

with db_session:
    cat_ages = select(c.age for c in Cat)  # Selecting only the age
    for age in cat_ages:
        print("\n----------------------------------------------------------------------\n")
        print(age)
        print("\n----------------------------------------------------------------------\n")


# ### Exercise 11

# In[53]:


with db_session:
    young_cats = select(c for c in Cat if c.age < 4)
    print("\n----------------------------------------------------------------------\n")
    for cat in young_cats:
        print(f"Name: {cat.name}, Age: {cat.age}")
    print("\n----------------------------------------------------------------------\n")


# ### Exercise 12

# In[56]:


with db_session:

    current_cat = Cat.get(name="Kira")
    if current_cat:
        current_cat.age += 1
        current_cat.color = 'New Color'
        # Commit changes at the end of the db_session block
    else:
        print("Cat named Kira not found.")


# ### Exercise 13

# In[57]:


# Deleting using the Result Object:
with db_session:
    cat_to_delete = Cat.get(name="Max")
    if cat_to_delete:
        cat_to_delete.delete()  # Delete the record


# Deleting using a Delete Generator Expression:
from pony.orm import delete
with db_session:
    delete(c for c in Cat if c.name == "Oliver")


# In[ ]:




