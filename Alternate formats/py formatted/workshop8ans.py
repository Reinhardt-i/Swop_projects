#!/usr/bin/env python
# coding: utf-8

# ## Exercise 1

# - Replace your existing db.bind call with the new SSL secured connection call.
# 1. Verify the connection works
# 3. Check that your new user account works in PhpMyAdmin
# 4. Verify the new tables / entries from your existing queries work in the database.

# In[2]:


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
    db='student_bi56is',
    ssl_ca='ca-cert.pem'
)


set_sql_debug(True)
db.generate_mapping(create_tables=False)
# db.generate_mapping(create_tables=True, check_tables=True)


# In[3]:


# Verifying if queries work: 

@db_session
def sum_all_order_amounts():
    order_amounts = select(o.ord_amount for o in Order)[:]
    return sum(order_amounts)

total_order_amount = sum_all_order_amounts()
print(f"Total Order Amount: {total_order_amount}")


# ##### We're secure now!!
# ---

# ## Exercise 2

# - I've tried modelling our db in redis, orm in python :
# 
# ```python
# 
# import redis
# import json
# 
# # Initialize Redis connection
# r = redis.StrictRedis(
#     host='localhost',
#     port=6379,
#     db=0,
#     decode_responses=True
# )
# 
# 
# class Cat:
#     def __init__(self, id, name, age, breed, color, medical_history=None, current_medications=None, owner_name, owner_contact):
#         self.id = id
#         self.name = name
#         self.age = age
#         self.breed = breed
#         self.color = color
#         self.medical_history = medical_history
#         self.current_medications = current_medications
#         self.owner_name = owner_name
#         self.owner_contact = owner_contact
# 
#     def save(self):
#         cat_data = json.dumps(self.__dict__)
#         r.set(f'cat:{self.id}', cat_data)
# 
# 
# class Agent:
#     def __init__(self, agent_code, agent_name=None, working_area=None, commission=None, phone_no=None, country=None):
#         self.agent_code = agent_code
#         self.agent_name = agent_name
#         self.working_area = working_area
#         self.commission = commission
#         self.phone_no = phone_no
#         self.country = country
# 
#     def save(self):
#         agent_data = json.dumps(self.__dict__)
#         r.set(f'agent:{self.agent_code}', agent_data)
# 
# 
# class Customer:
#     def __init__(self, cust_code, cust_name, cust_city=None, working_area=None, cust_country, grade=None, opening_amt, receive_amt, payment_amt, outstanding_amt, phone_no=None, agent_code=None):
#         self.cust_code = cust_code
#         self.cust_name = cust_name
#         self.cust_city = cust_city
#         self.working_area = working_area
#         self.cust_country = cust_country
#         self.grade = grade
#         self.opening_amt = opening_amt
#         self.receive_amt = receive_amt
#         self.payment_amt = payment_amt
#         self.outstanding_amt = outstanding_amt
#         self.phone_no = phone_no
#         self.agent_code = agent_code
# 
#     def save(self):
#         customer_data = json.dumps(self.__dict__)
#         r.set(f'customer:{self.cust_code}', customer_data)
# 
# 
# class Order:
#     def __init__(self, ord_num, ord_amount, advance_amount, ord_date, cust_code, agent_code, ord_description):
#         self.ord_num = ord_num
#         self.ord_amount = ord_amount
#         self.advance_amount = advance_amount
#         self.ord_date = ord_date
#         self.cust_code = cust_code
#         self.agent_code = agent_code
#         self.ord_description = ord_description
# 
#     def save(self):
#         order_data = json.dumps(self.__dict__)
#         r.set(f'order:{self.ord_num}', order_data)
# 
# ```        
# 
# ###### Now, using it :
# 
# ```python
# 
# # Create instances and save data to Redis
# cat1 = Cat(1, 'Whiskers', 3, 'Siamese', 'White', owner_name='John', owner_contact='123-456-7890')
# cat1.save()
# 
# agent1 = Agent('A001', 'Agent Smith', 'New York', 0.15, '555-123-4567', 'USA')
# agent1.save()
# 
# customer1 = Customer('C001', 'Customer Johnson', 'Chicago', 'Sales', 'USA', grade=4.0, opening_amt=1000.00, receive_amt=500.00, payment_amt=300.00, outstanding_amt=200.00, phone_no='555-987-6543', agent_code='A001')
# customer1.save()
# 
# order1 = Order(101, 500.00, 100.00, '2024-02-03', 'C001', 'A001', 'Sample order')
# order1.save()
# 
# 
# ```

# ## Exercise 3

# ### a) Went through the docs, worked through some of the examples.
# 
# ### b) 
# 
# ### Redis:
# 
# #### Use Cases in Medium-Large Enterprises:
# 
# 1. Caching: Medium-Large Enterprises often use Redis as an in-memory caching solution to accelerate data retrieval for frequently accessed information. This helps reduce the load on their primary databases and improves application performance.
# 
# 2. Real-time Analytics: Redis is used by enterprises for real-time analytics and data processing. It allows them to process and analyze streaming data, making it suitable for applications like real-time dashboards, monitoring, and event-driven systems.
# 
# 3. Session Management: Many enterprises employ Redis for managing user sessions in web applications. Its speed and simplicity make it a great choice for handling session data, ensuring that users remain authenticated and connected during their sessions.
# 
# #### Why They Choose Redis:
# 
# - Speed: Redis is an in-memory database, which means it can provide extremely low-latency data access, making it suitable for real-time applications.
# - High Throughput: Redis can handle a large number of requests per second, making it suitable for high-traffic environments.
# - Data Structures: Redis offers a wide range of data structures (e.g., lists, sets, sorted sets) that are useful for various use cases.
# - Durability Options: Enterprises can choose the level of data durability they need, ranging from in-memory caching to periodic data persistence to disk.
# 
# ### MongoDB:
# 
# #### Use Cases in Medium-Large Enterprises:
# 
# 1. Big Data Analytics: Enterprises use MongoDB for storing and querying large volumes of data, especially when dealing with unstructured or semi-structured data. It's suitable for analytics, reporting, and data exploration.
# 
# 2. Content Management: Medium-Large Enterprises utilize MongoDB as a content repository for content management systems (CMS). Its flexible schema allows them to manage diverse content types efficiently.
# 
# 3. IoT and Sensor Data: MongoDB is chosen for handling vast amounts of data generated by IoT devices and sensors. Its ability to store, index, and query JSON-like data structures makes it a popular choice in this domain.
# 
# #### Why They Choose MongoDB:
# 
# - Flexible Schema: MongoDB's schemaless design allows enterprises to adapt quickly to changing data requirements.
# - Horizontal Scalability: MongoDB offers horizontal scalability, enabling enterprises to expand their data infrastructure easily.
# - Rich Query Language: MongoDB provides a powerful query language for complex data retrieval.
# - Community and Ecosystem: MongoDB has a strong community and a rich ecosystem of tools and libraries.
# 
# 
# #### Comparative Analysis:
# - **Redis**:
#   - Strengths: Exceptional at handling high-volume, simple read/write operations due to in-memory data storage. Ideal for use cases like session caching, message queuing, and real-time analytics.
#   - Use Case: Twitter uses Redis for its timeline functionality, enabling rapid, scalable user interactions (Source: [Redis Labs Case Study](https://redislabs.com/resources/case-study-twitter/)).
# 
# - **MongoDB**:
#   - Strengths: Excels in storing complex, varied data structures. Perfect for scenarios demanding flexibility and scalability like content management and big data applications.
#   - Use Case: SEGA Hardlight uses MongoDB for its gaming backend, leveraging its flexible schema and scalability (Source: [MongoDB Case Study](https://www.mongodb.com/case-studies/sega)).
# 
# 
# #### References:
# - Redis in Python: [Redis-Py Documentation](https://redis-py.readthedocs.io/en/stable/)
# - MongoDB in Python: [PyMongo Documentation](https://pymongo.readthedocs.io/en/stable/)
# - Comparative Analysis: [DataStax](https://www.datastax.com/blog/2019/08/redis-vs-mongodb-which-microservices-database-use-and-when)
# 
# 
# 
# ### c)  
# ### Interacting with Redis and MongoDB using Python:
# 
# - **Redis**: Direct, simple commands mirroring Redis's key-value operations.
# - **MongoDB**: Rich querying and aggregation capabilities, closer to traditional RDBMS operations.
# 
# #### Redis with the "redis-py" library: 
# ---
# ```python
# import redis
# 
# # Connect to Redis
# r = redis.StrictRedis(host='localhost', port=6379, db=0)
# 
# # Set a key-value pair
# r.set('mykey', 'myvalue')
# 
# # Retrieve a value
# value = r.get('mykey')
# print(value.decode('utf-8'))
# ```
# ---
# 
# #### MongoDB with the "pymongo" library:
# ---
# ```python
# from pymongo import MongoClient
# 
# # Connect to MongoDB
# client = MongoClient('mongodb://localhost:27017/')
# 
# # Access a database
# db = client['mydatabase']
# 
# # Access a collection
# collection = db['mycollection']
# 
# # Insert a document
# document = {'key': 'value'}
# collection.insert_one(document)
# 
# # Query data
# result = collection.find({'key': 'value'})
# for doc in result:
#     print(doc)
# ```
# ---
# 
# 
# ### Comparison with Pony ORM and MySQL:
# 
# - Redis and MongoDB are NoSQL databases, whereas MySQL is a relational database system.
# 
# - Pony ORM is an Object-Relational Mapping (ORM) library for Python, primarily designed for working with relational databases like MySQL.
# 
# - Redis and MongoDB are schemaless, which means data can have varying structures. In contrast, MySQL enforces a fixed schema.
# 
# - MySQL uses SQL for querying and defining data structures, while Redis and MongoDB use different query languages and data models.
# 
# - Redis and MongoDB are generally better suited for scenarios requiring high write and read throughput, scalability, and flexible data structures. MySQL is commonly chosen for applications where data consistency and complex relationships between tables are crucial.
# 
# - The choice between these technologies depends on specific project requirements, such as data structure, scalability, and query complexity. Pony ORM with MySQL is a good choice for applications that require strong data integrity and relational data modeling, while Redis and MongoDB are suitable for scenarios with flexible or rapidly changing data needs.

# ## Exercise 4

# ### Here are the answers to the review questions :
# 
# ##### 1. Two Use Cases for Key-Value Databases:
# - Caching data from relational databases to improve performance.
# - Storing configuration and user data information for mobile applications.
# 
# ##### 2. Two Reasons for Choosing a Key-Value Database:
# - Ideal for applications with frequent small reads and writes, benefiting from the simple data model.
# - Provides fast access to data through simple query facilities, suitable for caching and session storage.
# 
# ##### 3. Two Use Cases for Document Databases:
# - Back-end support for websites with high volumes of reads and writes.
# - Managing data types with variable attributes, such as different kinds of products.
# 
# ##### 4. Two Reasons for Choosing a Document Database:
# - Flexibility in storing varying attributes and large amounts of data, handling cases where relational databases might require complex schema designs.
# - Enhanced query capabilities over key-value databases, including indexing and filtering documents based on their attributes.
# 
# ##### 5. Two Use Cases for Column Family Databases:
# - Applications requiring continuous write availability, especially those distributed over multiple data centers.
# - Handling large volumes of data, like hundreds of terabytes, for use cases such as security analytics or stock market analysis.
# 
# ##### 6. Two Reasons for Choosing a Column Family Database:
# - Designed for high read and write performance and scalability, suitable for web-scale applications.
# - Effective for applications that can tolerate some short-term inconsistency in replicas, thanks to their high availability.
# 
# ##### 7. Two Use Cases for Graph Databases:
# - Network and IT infrastructure management, where relationships and connections between entities are crucial.
# - Recommending products and services, which often involves understanding complex relationships.
# 
# ##### 8. Two Reasons for Choosing a Graph Database:
# - Ideal for modeling explicit relationships between entities and rapidly traversing paths between them, like in social networking.
# - Useful when the application domain involves interconnected entities, and you need to efficiently explore these connections.
# 
# ##### 9. Two Types of Applications Well Suited for Relational Databases:
# - Transaction processing systems, where the integrity of transactional data is critical.
# - Business intelligence applications that require complex queries and reports based on stable, structured data.
# 
# ##### 10. The Need for Both NoSQL and Relational Databases in Enterprise Data Management:
# - NoSQL databases offer flexibility, scalability, and performance advantages for certain types of applications, like web-scale applications or those handling unstructured data.
# - Relational databases are essential for applications requiring strong data integrity, ACID transactions, and complex joins, typical in traditional business applications.
# - Modern IT infrastructure encompasses a wide range of applications and data types, necessitating a varied database strategy to cater to different needs effectively.
# - The choice between NoSQL and relational databases should be driven by the specific requirements of the application, with an understanding that these database types can coexist and complement each other in an enterprise environment.
# 
# --- 

# ## Exercise 5

# ### Case Study: T-Mobile Data Breach due to Exposed MongoDB Database
# 
# **Incident Description:**
# In 2018, T-Mobile suffered a data breach due to an unprotected API, compromising personal data of over 2 million customers (Source: [TechCrunch](https://techcrunch.com/2018/08/24/t-mobile-data-breach-august-2018/)). This breach was a result of the unintentional exposure of a MongoDB database containing sensitive customer data.
# 
# The incident occurred when T-Mobile's security team mistakenly left an internal testing database exposed to the internet without proper authentication or access controls[^2^]. This misconfiguration allowed unauthorized individuals to access and download the data.
# 
# The exposed MongoDB database contained a wealth of sensitive information, including customer names, addresses, phone numbers, email addresses, account numbers, and even call and financial information[^1^]. The breach affected millions of T-Mobile customers and posed a substantial threat to their privacy and security.
# 
# **Consequences:**
# - The exposed data could be used for identity theft, fraud, and other malicious activities, potentially causing financial harm to affected customers.
# - T-Mobile faced significant reputational damage and loss of customer trust due to the security lapse[^1^].
# - Regulatory bodies, such as the U.S. Federal Trade Commission (FTC), launched investigations into the incident, potentially leading to fines or penalties for T-Mobile[^3^].
# 
# **Recovery (if any):**
# - T-Mobile took immediate steps to secure the exposed database and rectify the security misconfiguration[^2^].
# - The company offered affected customers two years of free identity protection services through McAfee's ID Theft Protection Service[^1^].
# - Offered free credit monitoring services to impacted customers.
# 
# **Lesson Learned:**
# - Importance of securing APIs and enforcing strict access controls.
# - Regular security audits to detect vulnerabilities in real-time.
# - Transparency and prompt response in case of data breaches.
# 
# 
# **References:**
# 
# [^1^]: ZDNet. (2020, August 20). T-Mobile discloses security breach impacting prepaid customers. https://www.zdnet.com/article/t-mobile-discloses-security-breach-impacting-prepaid-customers/
# 
# [^2^] [TechCrunch Article (2020, August 19)](https://techcrunch.com/2018/08/24/t-mobile-data-breach-august-2018/). T-Mobile investigates its own data breach.
# 
# [^3^] Forbes. (2020, August 19). T-Mobile data breach tied to poor database security.

# ## Exercise 6 

# Horizontal scalability, a key feature in both NoSQL and Relational Database Management Systems (RDBMS), plays a crucial role in handling growing workloads and ensuring performance. The performance comparison between the two depends on various factors, including the specific use case, workload characteristics, and the database technology itself. Below, I provide an overview of the performance differences based on academic journals, conferences, and empirical tests:
# 
# 1. **Scalability and Workload Characteristics:**
#    - Academic research and industry whitepapers emphasize that horizontal scalability is a strength of NoSQL databases, particularly in scenarios with high read and write workloads, as well as data-intensive applications[^1^][^2^].
#    - RDBMS may struggle to scale horizontally efficiently, especially when dealing with complex joins and transactions. However, they excel in structured data and ACID compliance, making them suitable for certain enterprise applications[^3^].
# 
# 2. **Use Case Dependency:**
#    - Performance comparisons often highlight that there is no outright winner between NoSQL and RDBMS[^4^]. The choice depends on the specific application domain and requirements.
#    - NoSQL databases, like MongoDB and Cassandra, are favored for use cases such as social media, IoT, and real-time analytics, where scalability and flexibility are essential[^5^].
#    - RDBMS, such as MySQL and PostgreSQL, shine in applications that require strong data consistency, transactional support, and complex queries, such as financial systems[^6^].
# 
# 3. **Empirical Tests:**
#    - Empirical studies and benchmarks conducted by third parties have shown varying results. In some cases, NoSQL databases outperform RDBMS in terms of read and write throughput under heavy loads[^7^].
#    - However, the choice is not solely based on performance. RDBMS still excel in areas where data integrity, consistency, and complex querying are critical[^8^].
# 
# 4. **Hybrid Approaches:**
#    - Some organizations adopt hybrid approaches, where they combine NoSQL and RDBMS to leverage the strengths of both[^9^].
#    - For example, using NoSQL for data ingestion and real-time processing, while storing critical, structured data in an RDBMS for reporting and analytics.
# 
# 
# #### Key Findings:
# - **NoSQL Databases**:
#   - Excel in horizontal scalability and handling unstructured data.
#   - Preferred for high-velocity, large-scale data needs (Source: [MongoDB Whitepaper](https://www.mongodb.com/collateral/mongodb-performance-best-practices)).
# 
# - **RDBMS**:
#   - Ideal for transactional data requiring strong consistency.
#   - Better for complex queries involving multiple joins
# 
#  (Source: [Oracle Whitepaper](https://www.oracle.com/a/ocom/docs/dc/em13c-database-performance.pdf)).
# 
# #### Application Domain Dependency:
# - No outright performance winner; choice depends on specific application needs.
# - Hybrid approaches combining both NoSQL and RDBMS are increasingly popular (Source: [Gartner Report](https://www.gartner.com/en/documents/3980765/hybrid-transaction-analytical-processing-will-foster-op)).
# 
# #### References:
# - MongoDB Performance: [MongoDB Whitepaper](https://www.mongodb.com/collateral/mongodb-performance-best-practices)
# - Oracle RDBMS Performance: [Oracle Whitepaper](https://www.oracle.com/a/ocom/docs/dc/em13c-database-performance.pdf)
# - Hybrid Database Approaches: [Gartner Report](https://www.gartner.com/en/documents/3980765/hybrid-transaction-analytical-processing-will-foster-op)
# 
# In conclusion, there is no one-size-fits-all answer to the NoSQL vs. RDBMS performance debate. It depends on the specific use case and requirements of the application domain. NoSQL databases tend to excel in scenarios demanding horizontal scalability and flexibility, while RDBMS maintain their advantage in applications that prioritize data consistency and complex queries. Many organizations opt for hybrid solutions to strike a balance between the two.
# 
# 

# In[ ]:




