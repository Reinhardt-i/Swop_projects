#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1

# #### Analysis of a Company's Data Trends
# 
# **Company Selected:** Amazon
# 
# 1. **Data Generation:** Amazon generates an enormous amount of data. Based on an estimated analysis, Amazon generates approximately 50 terabytes of data daily. This includes customer transactions, site traffic, and interaction data.
# 
# 2. **Day-to-Day Data Usage:** Amazon uses various types of data in its daily operations:
#     - **Transaction Data:** For processing and tracking orders.
#     - **Customer Interaction Data:** To improve user experience and interface design.
#     - **Analysis Data:** Consumer behavior data is used for market analysis and predictive analytics.
# 
# 3. **Data Usage:** 
#     - **Personalized Recommendations:** Analyzing purchase history to recommend products.
#     - **Inventory Management:** Predictive analytics for stock management.
#     - **Customer Service:** Enhancing user experience based on feedback and interaction data.
# 
# 4. **Data Sources:**
#     - **Customer Interactions:** Data from website visits, app usage, and Alexa interactions.
#     - **Transactional Data:** From online purchases and third-party sellers.
#     - **External Data Sources:** Market trends, demographic data, etc.
# 
# 5. **Data Collection Methods:**
#     - **Automated Tracking:** Of user interactions on their platforms.
#     - **Direct Input:** From users during transactions and account settings.
#     - **Third-Party Data Providers:** For supplemental market and demographic information.
# 
# 6. **Data Permissions:**
#     - **User Agreements:** Customers agree to data collection through terms of service.
#     - **Data Partnerships:** With third-party sellers and service providers.
#     - **Regulatory Compliance:** Adhering to data protection laws like GDPR.
# 
# ---

# ### Exercise 2

# #### Data Mining Use-Cases
# 
# 1. **Log Analytics:**
#    - **Company:** Splunk
#    - **Value Extraction:** Splunk uses log analytics for monitoring, searching, and analyzing machine-generated big data. This helps in IT operations, security, and business analytics.
#    - **Technology:** Big data analysis tools, machine learning.
# 
# 2. **Commerce:**
#    - **Company:** Walmart
#    - **Value Extraction:** Utilizes data mining for inventory management, optimizing supply chain, and customer behavior analysis to improve sales.
#    - **Technology:** Big data platforms, predictive analytics tools.
# 
# 3. **Recommendation Systems:**
#    - **Company:** Netflix
#    - **Value Extraction:** Netflix uses data mining to personalize content recommendations based on user viewing history and preferences.
#    - **Technology:** Machine learning algorithms, cloud computing.
# 
# 4. **Fault Detection/Prediction:**
#    - **Company:** GE Aviation
#    - **Value Extraction:** GE uses data mining for predictive maintenance of aircraft engines, reducing downtime and maintenance costs.
#    - **Technology:** IoT sensors, predictive analytics software.
# 
# 5. **Fraud Detection:**
#    - **Company:** PayPal
#    - **Value Extraction:** PayPal employs data mining to detect and prevent fraudulent transactions, enhancing security for its users.
#    - **Technology:** AI and machine learning algorithms, real-time analytics.
# 
# **Additional Use-Case:** 
#    - **Healthcare Predictive Analytics:** Companies like IBM use data mining in healthcare for predictive analysis of patient outcomes, treatment optimization, and disease spread modeling.
# 
# ---

# ### Exercise 3

# In[8]:


import pandas as pd

titanic_df = pd.read_csv('titanic.csv')
print(titanic_df.head())


# ---
# 
# #### Titanic Dataset Analysis
# 
# Based on the initial data load and the provided Data Dictionary, here are the data types for each attribute:
# 
# - `Survived`: Boolean (0 or 1, representing a binary categorical variable)
# - `Name` : Textual (String)
# - `Pclass`: Categorical (ordinal, as the classes have a natural order)
# - `Sex`: Categorical (nominal, as there is no order between male and female)
# - `Age`: Numeric (continuous, as age can be measured on a continuum)
# - `SibSp`: Numeric (discrete, as the number of siblings/spouses is countable)
# - `Parch`: Numeric (discrete, similar to SibSp)
# - `Ticket`: Categorical (nominal, as ticket numbers are arbitrary labels)
# - `Fare`: Numeric (continuous, as fare can take any value within a range)
# - `Cabin`: Categorical (nominal, as cabin numbers are unique identifiers)
# - `Embarked`: Categorical (nominal, representing ports as labels)
# 
# ---

# ### Exercise 4

# In[14]:


min_fare = titanic_df['Fare'].min()
max_fare = titanic_df['Fare'].max()
print(f"\nMinimum Fare: {min_fare}, Maximum Fare: {max_fare}\n")

fare_data_type_python = titanic_df['Fare'].dtype
print(f"Python Data Type for Fare: {fare_data_type_python}\n")

class_fare_summary = titanic_df.groupby('Pclass')['Fare'].agg(['min', 'max']).reset_index()
print(class_fare_summary)


# 
# #### 4.1) Minimum and Maximum Fare Calculation
# 
# The following Python code calculates the minimum and maximum fare in the dataset.
# 
# ```python
# 
# min_fare = titanic_df['Fare'].min()
# max_fare = titanic_df['Fare'].max()
# print(f"Minimum Fare: {min_fare}, Maximum Fare: {max_fare}")
# 
# ```
# 
# - Minimum fare in the dataset: 0.0
# - Maximum fare in the dataset: 512.3292
# 
# 
# #### 4.2) Data Type of Fare Attribute
# 
# ```python
# 
# fare_data_type_python = titanic_df['Fare'].dtype
# print(f"Python Data Type for Fare: {fare_data_type_python}")
# 
# ```
# - Python data type: `float` (since it can contain decimal values)
# - Data Science data type: Continuous numerical variable (it can take on any value within a range)
# 
# 
# #### 4.3) Fare as a Range Descriptor
# 
# - Fare attribute as a range descriptor: [0, 512.33]
# 
# 
# #### 4.4. Degree of Precision for Fare
# 
# For the fare attribute, a good degree of precision would be two decimal places. This precision level is standard for financial amounts, balancing accuracy and readability.
# 
# 
# #### 4.5) Min and Max Fare by Passenger Class (Pclass)
# 
# The code below groups the data by 'Pclass' and calculates the minimum and maximum fare for each class.
# 
# ```python
# 
# class_fare_summary = titanic_df.groupby('Pclass')['Fare'].agg(['min', 'max']).reset_index()
# print(class_fare_summary)
# 
# ```
# 
# Here are the results:
# 
#    Pclass  min       max
# 0       1  0.0  512.3292
# 1       2  0.0   73.5000
# 2       3  0.0   69.5500
# 
# - 1st Class: min fare 0.0, max fare 512.3292
# - 2nd Class: min fare 0.0, max fare 73.5000
# - 3rd Class: min fare 0.0, max fare 69.5500
# 
# 
# 
# Apologies for missing that part. Let's address question 4.6 regarding the `Name` attribute from the Titanic dataset:
# 
# #### 4.6) Consideration of the Name Attribute
# 
# The `Name` attribute in the dataset is a combination of titles, first names, potential middle names, last names, and sometimes additional information such as aliases or marital status. To express this in a structured form, we could break down the attribute into several parts:
# 
# - *Title*: This could include titles such as Mr, Mrs, Miss, Dr, Prof, etc., and is typically indicative of gender, marital status, and social status.
# - *First Name*: The given name of the passenger.
# - *Middle Name(s)*: Any additional given names.
# - *Last Name*: The family name or surname of the passenger.
# - *Additional Information*: This might include aliases, nicknames, or in the case of married women, their maiden names may be included in parentheses.
# 
# In a structured data format, these could be represented as separate columns to facilitate analysis and provide clarity. For instance, determining family relationships would be easier with a dedicated 'Last Name' column, and analysis of social status or gender could be aided by a 'Title' column.
# 
# ---

# ### Exercise 5

# #### Some data dictionaries from publicly available datasets for comparison:
# 
# 1. **CIBMTR (Center for International Blood and Marrow Transplant Research) Datasets** :
#    - **Data Dictionary**: Describes the data for various studies related to blood and marrow transplants. Each dataset focuses on specific aspects of transplant outcomes, such as the impact of pre-transplant induction, the pathogenicity of HLA class I alleles, and the impact of pre-apheresis health-related quality of life.
# 
# 2. **OpenFEMA Datasets** :
#    - **Data Dictionary**: Provides metadata for datasets related to emergency management, individual assistance, preparedness, and alerts. The data dictionaries describe each field in datasets that cover areas such as disaster declarations, emergency management performance grants, and housing assistance program data.
# 
# 3. **PLCO (Prostate, Lung, Colorectal, and Ovarian) Cancer Screening Trial Datasets** :
#    - **Data Dictionary**: Offers detailed descriptions for datasets related to the PLCO cancer screening trial. It provides comprehensive data for colorectal cancer screening, incidence, mortality analyses, and related diagnostic and treatment information.
# 
# #### Comparing these data dictionaries to the Titanic dataset and the games example from the lecture slides:
# 
# - **1. Level of Description**: The CIBMTR and PLCO data dictionaries provide a very detailed level of description, which is useful for researchers and healthcare professionals to understand the nuances of the data. The OpenFEMA data dictionary also offers detailed descriptions, essential for emergency management and disaster response analysis. The Titanic dataset provides a basic description useful for educational data analysis purposes.
# 
# - **2. Accuracy**: All data dictionaries appear to provide accurate and precise definitions for each variable, which is crucial for any data analysis to ensure the reliability of the results.
# 
# - **3. Usefulness for Multiple Roles**:
#    - **a.) Analysts** would benefit from the detailed descriptions and specific variable definitions provided in all data dictionaries to conduct thorough and accurate analyses.
#    - **b.) Designers** can use the structure and organization of these data dictionaries to create interfaces or visualizations that accurately represent the underlying data.
#    - **c.) Developers** need detailed data dictionaries like these to understand how to manipulate and process the data effectively, ensuring the correct handling of data types, missing values, and understanding the relationships between different data points.
# 
# In conclusion, a good data dictionary should provide the level of detail and accuracy that aligns with the needs of its primary users, which may vary from dataset to dataset. The examples we've seen cover a wide range of uses, from healthcare and emergency management to historical data analysis, each with a tailored level of detail and specificity.
# 
# 
# ---

# ### Exercise 6

# To improve the data dictionary for the Titanic dataset and make it more complete and useful, I will use the information from Exercise 3 and the given dataset. Here is an enhanced version of the data dictionary with justifications for each modification or addition:
# 
# ### Enhanced Data Dictionary for the Titanic Dataset
# 
# 1. **PassengerId**: 
#    - **Type**: Numeric (integer).
#    - **Description**: A unique identifier for each passenger.
#    - **Justification**: Including PassengerId helps in uniquely identifying records, which is essential for data management and analysis.
# 
# 2. **Survived**: 
#    - **Type**: Boolean (integer encoded as 0 or 1).
#    - **Description**: Indicates whether the passenger survived (1) or did not survive (0).
#    - **Justification**: Boolean type simplifies the understanding of survival status, making it clear and binary.
# 
# 3. **Pclass**: 
#    - **Type**: Categorical (ordinal).
#    - **Description**: The passenger class, with 1 being the highest class and 3 the lowest class.
#    - **Justification**: Treating Pclass as ordinal acknowledges the inherent order and social hierarchy present in the class system.
# 
# 4. **Name**: 
#    - **Type**: Textual (string).
#    - **Description**: The full name of the passenger, potentially including titles, first names, middle names, last names, and additional information.
#    - **Justification**: Expanding the description to include the structure of names can be beneficial for detailed analysis, such as family ties or social status.
# 
# 5. **Sex**: 
#    - **Type**: Categorical (nominal).
#    - **Description**: The gender of the passenger (male or female).
#    - **Justification**: Categorical treatment of gender allows for straightforward demographic analysis.
# 
# 6. **Age**: 
#    - **Type**: Numeric (continuous).
#    - **Description**: The age of the passenger. Can be fractional for children less than one year old.
#    - **Justification**: Treating age as a continuous variable allows for more nuanced analysis, like age distributions and correlations with survival.
#    
# 7. **AgeGroup**: 
#    - **Type**: Categorical (ordinal).
#    - **Description**: A categorical representation of age, grouping passengers into predefined age ranges (e.g., child, teenager, adult, senior).
#    - **Justification**: Grouping ages into categories can simplify analysis and highlight age-related trends in survival rates.
#    
# 8. **IsAlone**: 
#    - **Type**: Boolean (integer encoded as 0 or 1).
#    - **Description**: Indicates whether the passenger is traveling alone (0 if with family, 1 if alone). This can be derived from the `FamilySize` column.
#    - **Justification**: Traveling alone or with family can impact a passenger's survival chances and decision-making during the disaster.
# 
# 9. **SibSp**: 
#    - **Type**: Numeric (discrete).
#    - **Description**: The number of siblings or spouses aboard.
#    - **Justification**: Counting siblings and spouses provides insights into family sizes and the impact of companions on survival rates.
# 
# 10. **Parch**: 
#    - **Type**: Numeric (discrete).
#    - **Description**: The number of parents or children aboard.
#    - **Justification**: Similar to SibSp, provides understanding of family dynamics on board.
# 
# 11. **Ticket**: 
#    - **Type**: Categorical (nominal).
#    - **Description**: Ticket number.
#    - **Justification**: While nominal, ticket numbers can be useful for identifying groups traveling together or ticket patterns.
# 
# 12. **Fare**: 
#    - **Type**: Numeric (continuous).
#    - **Description**: The ticket fare paid by the passenger.
#    - **Justification**: As a continuous variable, fare analysis can yield insights into economic status and its correlation with survival.
# 
# 13. **Cabin**: 
#    - **Type**: Categorical (nominal).
#    - **Description**: Cabin number where the passenger stayed.
#    - **Justification**: Cabin location can be a significant factor in survival analysis and understanding passenger demographics.
#    
# 14. **Deck**: 
#    - **Type**: Categorical (nominal).
#    - **Description**: Extracted from the `Cabin` column, representing the deck on which the cabin is located (e.g., A, B, C).
#    - **Justification**: The deck location can be crucial for survival analysis, as it might correlate with the accessibility of lifeboats and evacuation routes.
# 
# 15. **Embarked**: 
#    - **Type**: Categorical (nominal).
#    - **Description**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).
#    - **Justification**: Provides geographical context and can be linked to passenger demographics and survival patterns.
# 
# 
# ### Additional Considerations
# 
# - **Missing Data**: Addressing the handling of missing data for attributes like Age, Cabin, and Embarked.
# - **Data Consistency**: Ensuring consistent formats, especially for textual data like names and tickets.
# - **Derived Attributes**: Considering the creation of derived attributes like family size (SibSp + Parch) for more profound analytical insights.
# 
# By enhancing the data dictionary with detailed descriptions, types, and justifications, the Titanic dataset becomes more valuable for comprehensive analysis, allowing for richer insights and more informed conclusions.
# 
# 
# ---

# In[ ]:




